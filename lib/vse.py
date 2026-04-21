import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

import lib.utils as utils
from lib.cross_net import CrossSparseAggrNet_v2
from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import loss_select


logger = logging.getLogger(__name__)


class VSEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.img_enc = get_image_encoder(opt)
        self.txt_enc = get_text_encoder(opt)
        self.criterion = loss_select(opt, loss_type=opt.loss)
        self.Eiters = 0
        self.cross_net = CrossSparseAggrNet_v2(opt)
        self.current_epoch = 0

    def set_current_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def freeze_backbone(self):
        self.img_enc.freeze_backbone()
        self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self):
        self.img_enc.unfreeze_backbone()
        self.txt_enc.unfreeze_backbone()

    def set_max_violation(self, max_violation=True):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def _reasonableness_scale(self):
        if not bool(getattr(self.opt, 'use_reasonable_refine', 0)):
            return 0.0
        start = int(getattr(self.opt, 'reasonable_start_epoch', 9))
        warm = max(1, int(getattr(self.opt, 'reasonable_warmup_epochs', 6)))
        e = int(self.current_epoch)
        if e < start:
            return 0.0
        if e >= start + warm:
            return 1.0
        return float(e - start + 1) / float(warm)

    def forward_emb(self, images, captions, lengths, long_captions=None, long_lengths=None):
        images = images.cuda()
        img_emb = self.img_enc(images)

        captions = captions.cuda()
        lengths = lengths.cuda()
        cap_emb = self.txt_enc(captions, lengths)

        long_cap_emb = None
        if long_captions is not None and long_lengths is not None:
            long_captions = long_captions.cuda()
            long_lengths = long_lengths.cuda()
            long_cap_emb = self.txt_enc(long_captions, long_lengths)

        if long_cap_emb is not None:
            return img_emb, cap_emb, lengths, long_cap_emb, long_lengths
        return img_emb, cap_emb, lengths

    def forward_sim(
        self,
        img_embs,
        cap_embs,
        cap_lens,
        long_cap_embs=None,
        long_cap_lens=None,
        enable_refine=None,
    ):
        return self.cross_net(
            img_embs,
            cap_embs,
            cap_lens,
            long_cap_embs,
            long_cap_lens,
            enable_refine=enable_refine,
        )

    def _gather_long_embeddings(self, long_cap_emb, long_lengths):
        if long_cap_emb is None or long_lengths is None:
            return None, None

        long_lengths = utils.concat_all_gather(long_lengths, keep_grad=False)
        max_long_len = int(long_lengths.max())
        if max_long_len > long_cap_emb.shape[1]:
            pad_long_emb = torch.zeros(
                long_cap_emb.shape[0],
                max_long_len - long_cap_emb.shape[1],
                long_cap_emb.shape[2],
                device=long_cap_emb.device,
                dtype=long_cap_emb.dtype,
            )
            long_cap_emb = torch.cat([long_cap_emb, pad_long_emb], dim=1)
        long_cap_emb = utils.all_gather_with_grad(long_cap_emb)
        return long_cap_emb, long_lengths

    def forward(
        self,
        images,
        captions,
        lengths,
        img_ids=None,
        warmup_alpha=1.0,
        long_captions=None,
        long_lengths=None,
    ):
        self.Eiters += 1

        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)

        long_cap_emb = None
        if long_captions is not None and long_lengths is not None:
            long_cap_emb = self.txt_enc(long_captions, long_lengths)

        new_img_emb = img_emb
        new_cap_emb = cap_emb
        new_long_cap_emb = long_cap_emb

        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)

            max_len = int(lengths.max())
            if max_len > new_cap_emb.shape[1]:
                pad_emb = torch.zeros(
                    new_cap_emb.shape[0],
                    max_len - new_cap_emb.shape[1],
                    new_cap_emb.shape[2],
                    device=new_cap_emb.device,
                    dtype=new_cap_emb.dtype,
                )
                new_cap_emb = torch.cat([new_cap_emb, pad_emb], dim=1)

            new_img_emb = utils.all_gather_with_grad(new_img_emb)
            new_cap_emb = utils.all_gather_with_grad(new_cap_emb)
            new_long_cap_emb, long_lengths = self._gather_long_embeddings(new_long_cap_emb, long_lengths)

        reason_scale = self._reasonableness_scale()

        patch_refine_start = int(getattr(self.opt, 'patch_refine_start_epoch', 3))
        enable_refine = bool(getattr(self.opt, 'use_reasonable_refine', 0)) and (
            self.current_epoch >= patch_refine_start
        )

        sim_out = self.forward_sim(
            new_img_emb,
            new_cap_emb,
            lengths,
            new_long_cap_emb,
            long_lengths,
            enable_refine=enable_refine,
        )

        if isinstance(sim_out, tuple):
            improved_sims, score_mask_all, aux = sim_out
        else:
            improved_sims = sim_out
            score_mask_all = None
            aux = {
                'necessity_mean': improved_sims.new_zeros([]),
                'exactness_mean': improved_sims.new_zeros([]),
                'pos_margin_mean': improved_sims.new_zeros([]),
                'neg_margin_mean': improved_sims.new_zeros([]),
            }

        align_loss = self.criterion(new_img_emb, new_cap_emb, img_ids, improved_sims) * warmup_alpha

        ratio_loss = align_loss.new_zeros([])
        #if score_mask_all is not None:
        #    ratio_loss = (score_mask_all.mean() - self.opt.sparse_ratio) ** 2

        necessity_loss = align_loss.new_zeros([])
        exact_loss = align_loss.new_zeros([])
        if reason_scale > 0.0:
            necessity_loss = F.relu(
                getattr(self.opt, 'necessity_margin', 0.0) - aux['necessity_mean']
            )
            exact_loss = F.relu(
                getattr(self.opt, 'exact_margin', 0.05) - aux['exactness_mean']
            )

        loss = align_loss
        if getattr(self.opt, 'use_ratio_loss', 0):
            loss = loss + getattr(self.opt, 'ratio_weight', 2.0) * ratio_loss

        loss = loss + reason_scale * (
            getattr(self.opt, 'cf_loss_weight', 0.02) * necessity_loss
            + getattr(self.opt, 'exact_loss_weight', 0.05) * exact_loss
        )

        stats = {
            'align_loss': float(align_loss.detach().item()),
            'ratio_loss': float(ratio_loss.detach().item()),
            'cf_loss': float(necessity_loss.detach().item()),
            'exact_loss': float(exact_loss.detach().item()),
            'reason_scale': float(reason_scale),
            'necessity_mean': float(aux['necessity_mean'].detach().item()),
            'exactness_mean': float(aux['exactness_mean'].detach().item()),
            'pos_margin_mean': float(aux['pos_margin_mean'].detach().item()),
            'neg_margin_mean': float(aux['neg_margin_mean'].detach().item()),
        }
        return loss, stats


def create_optimizer(opt, model):
    decay_factor = 1e-4
    cross_lr_rate = 1.0

    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]

    text_params_no_bert = []
    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)

    params_list = [
        {'params': text_params_no_bert, 'lr': opt.learning_rate},
        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.vision_proj.parameters(), 'lr': opt.learning_rate},
        {'params': model.cross_net.parameters(), 'lr': opt.learning_rate * cross_lr_rate},
        {'params': model.criterion.parameters(), 'lr': opt.learning_rate},
    ]

    optimizer = torch.optim.AdamW(
        params_list,
        lr=opt.learning_rate,
        weight_decay=decay_factor,
    )
    return optimizer


if __name__ == '__main__':
    pass
