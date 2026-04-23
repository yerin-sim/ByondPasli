import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.xttn import mask_xattn_one_text


def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

    def forward(self, tokens, attention_x, attention_y):
        B_v, L_v, C = tokens.size()
        score = attention_x + attention_y
        num_keep_token = math.ceil(L_v * self.sparse_ratio)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))
        selected_score = torch.gather(score, dim=1, index=keep_policy)

        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)
        return select_tokens, extra_token, score_mask, selected_score


class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, keep_policy=None):
        weight = self.weight(x)
        weight = weight.transpose(2, 1) * self.scale
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)
            weight = weight - (1 - keep_policy) * 1e10
        weight = F.softmax(weight, dim=2)
        x = torch.bmm(weight, x)
        return x


class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.hidden_dim = opt.embed_size
        self.num_patches = opt.num_patches
        self.sparse_ratio = opt.sparse_ratio
        self.aggr_ratio = opt.aggr_ratio
        self.attention_weight = opt.attention_weight
        self.ratio_weight = opt.ratio_weight
        self.keeped_patches = int(self.num_patches * self.aggr_ratio * self.sparse_ratio)
        self.sparse_net_cap = TokenSparse(embed_dim=self.hidden_dim, sparse_ratio=self.sparse_ratio)
        self.sparse_net_long = TokenSparse(embed_dim=self.hidden_dim, sparse_ratio=self.sparse_ratio)
        self.aggr_net = TokenAggregation(dim=self.hidden_dim, keeped_patches=self.keeped_patches)

    def _normalize_score(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _build_text_global_bank(self, text_embs_norm, text_lens):
        globals_ = []
        for i in range(len(text_lens)):
            n_word = int(text_lens[i])
            text_i = text_embs_norm[i, :n_word, :]
            globals_.append(F.normalize(text_i.mean(dim=0), dim=-1))
        return torch.stack(globals_, dim=0)

    def _mine_hardest_negative(self, image_global, positive_index, text_global_bank):
        if text_global_bank.size(0) <= 1:
            neg_idx = torch.zeros(image_global.size(0), dtype=torch.long, device=image_global.device)
            neg_text = text_global_bank[neg_idx]
            return neg_text, neg_idx

        sims = torch.matmul(image_global, text_global_bank.t())
        sims = sims.clone()
        sims[:, positive_index] = -1e4
        neg_idx = sims.argmax(dim=1)
        neg_text = text_global_bank[neg_idx]
        return neg_text, neg_idx

    def _mean_topk_lastdim(self, scores, topk=2, lengths=None):
        B, K, L = scores.shape
        if L == 0:
            return torch.zeros(B, K, device=scores.device, dtype=scores.dtype)

        k_eff = min(topk, L)
        if lengths is None:
            vals, _ = torch.topk(scores, k=k_eff, dim=-1)
            return vals.mean(dim=-1)

        mask = torch.arange(L, device=scores.device).view(1, 1, L) < lengths.view(B, 1, 1)
        masked_scores = scores.masked_fill(~mask, -1e4)
        vals, _ = torch.topk(masked_scores, k=k_eff, dim=-1)
        valid_counts = lengths.clamp(min=1, max=k_eff).view(B, 1).float()
        rank_mask = (torch.arange(k_eff, device=scores.device).view(1, 1, k_eff) < valid_counts.long().unsqueeze(-1)).to(scores.dtype)
        vals = vals * rank_mask
        return vals.sum(dim=-1) / valid_counts

    def _compute_word_necessity(self, sim_pw):
        B, K, N = sim_pw.shape
        if N == 0:
            return torch.zeros(B, K, device=sim_pw.device, dtype=sim_pw.dtype)

        if K == 1:
            owner_margin = sim_pw[:, 0, :].clamp_min(0.0)
            necessity = owner_margin.unsqueeze(1) / float(max(N, 1))
            return necessity

        top_vals, top_idx = torch.topk(sim_pw, k=2, dim=1)
        a1 = top_vals[:, 0, :]
        a2 = top_vals[:, 1, :]
        owner = top_idx[:, 0, :]
        necessity = torch.zeros(B, K, device=sim_pw.device, dtype=sim_pw.dtype)
        owner_margin = F.relu(a1 - a2)
        necessity.scatter_add_(1, owner, owner_margin)
        #necessity = necessity / float(max(N, 1))
        return necessity

    def _compute_redundancy(self, selected_tokens_norm):
        B, K, _ = selected_tokens_norm.shape
        if K <= 1:
            return torch.zeros(B, K, device=selected_tokens_norm.device, dtype=selected_tokens_norm.dtype)

        sim_pp = torch.bmm(selected_tokens_norm, selected_tokens_norm.transpose(1, 2))
        eye = torch.eye(K, device=selected_tokens_norm.device, dtype=torch.bool).unsqueeze(0)
        sim_pp = sim_pp.masked_fill(eye, -1e4)
        redundancy = sim_pp.max(dim=-1).values.clamp_min(0.0)
        return redundancy

    def _compute_refined_tokens(
        self,
        img_cls_emb,
        selected_tokens,
        extra_token,
        selected_score,
        pos_text_global,
        pos_text_expand,
        positive_index,
        text_global_bank,
        text_token_bank,
        text_lens_bank,
        enable_refine,
    ):
        B_v, K, C = selected_tokens.shape
        if K == 0:
            return selected_tokens, selected_score, None

        selected_tokens_norm = F.normalize(selected_tokens, dim=-1)
        selected_global = F.normalize(selected_tokens_norm.mean(dim=1), dim=-1)

        if pos_text_global.dim() == 1:
            pos_text_global = pos_text_global.unsqueeze(0).expand(B_v, -1)
        elif pos_text_global.dim() == 3:
            pos_text_global = pos_text_global.mean(dim=1)
        pos_text_global = F.normalize(pos_text_global, dim=-1)

        pos_len = pos_text_expand.size(1)
        sim_pos = torch.bmm(selected_tokens_norm, pos_text_expand.transpose(1, 2))
        necessity_score = self._compute_word_necessity(sim_pos)
        pos_patch_support = self._mean_topk_lastdim(sim_pos, topk=2, lengths=None)

        hard_neg_text, hard_neg_idx = self._mine_hardest_negative(selected_global, positive_index, text_global_bank)
        neg_text_expand = text_token_bank.index_select(0, hard_neg_idx)
        neg_text_lens = text_lens_bank.index_select(0, hard_neg_idx)
        sim_neg = torch.bmm(selected_tokens_norm, neg_text_expand.transpose(1, 2))
        #neg_patch_affinity = self._mean_topk_lastdim(sim_neg, topk=2, lengths=neg_text_lens)
        neg_patch_affinity = self._mean_topk_lastdim(sim_neg, topk=1, lengths=neg_text_lens)
        #exactness_score = pos_patch_support - neg_patch_affinity
        exactness_score = pos_patch_support - 0.35 * neg_patch_affinity

        redundancy_score = self._compute_redundancy(selected_tokens_norm)
        attn_prior = selected_score

        fused_score = (
            getattr(self.opt, 'score_attn_weight', 0.15) * self._normalize_score(attn_prior)
            + getattr(self.opt, 'score_nec_weight', 0.45) * self._normalize_score(necessity_score)
            + getattr(self.opt, 'score_exact_weight', 0.30) * self._normalize_score(exactness_score)
            - getattr(self.opt, 'score_red_weight', 0.10) * self._normalize_score(redundancy_score)
        )

        if enable_refine:
            keep_k = max(1, int(math.ceil(K * getattr(self.opt, 'refine_ratio', 0.85))))
            _, refine_idx = torch.topk(fused_score, k=keep_k, dim=1, largest=True, sorted=True)
            refined_tokens = torch.gather(selected_tokens, 1, refine_idx.unsqueeze(-1).expand(-1, -1, C))
            refined_score = torch.gather(fused_score, 1, refine_idx)
            refined_tokens_norm = F.normalize(refined_tokens, dim=-1)
            refined_pos_patch_support = torch.gather(pos_patch_support, 1, refine_idx)
            refined_neg_patch_affinity = torch.gather(neg_patch_affinity, 1, refine_idx)
            refined_global = F.normalize(refined_tokens_norm.mean(dim=1), dim=-1)
        else:
            refine_idx = None
            refined_tokens = selected_tokens
            refined_tokens_norm = selected_tokens_norm
            refined_score = fused_score
            refined_pos_patch_support = pos_patch_support
            refined_neg_patch_affinity = neg_patch_affinity
            refined_global = selected_global

        aggr_tokens = self.aggr_net(refined_tokens)
        keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)
        if self.has_cls_token:
            final_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
        else:
            final_tokens = keep_spatial_tokens
        final_tokens = F.normalize(final_tokens, dim=-1)

        if hard_neg_text.dim() == 1:
            hard_neg_text = hard_neg_text.unsqueeze(0).expand(B_v, -1)
        elif hard_neg_text.dim() == 3:
            hard_neg_text = hard_neg_text.mean(dim=1)
        hard_neg_text = F.normalize(hard_neg_text, dim=-1)

        meta = {
            'necessity_score': necessity_score.mean(dim=1),
            'exactness_score': exactness_score.mean(dim=1),
            'redundancy_score': redundancy_score.mean(dim=1),
            'selected_score': selected_score.mean(dim=1),
            'refined_score': refined_score.mean(dim=1),
            'hard_neg_idx': hard_neg_idx.float(),
            'refined_pos_mean': refined_pos_patch_support.mean(dim=1),
            'refined_neg_mean': refined_neg_patch_affinity.mean(dim=1),
            'refined_global_pos': (refined_global * pos_text_global).sum(dim=-1),
            'refined_global_neg': (refined_global * hard_neg_text).sum(dim=-1),
            'pos_word_count': torch.full((B_v,), float(pos_len), device=selected_tokens.device, dtype=selected_tokens.dtype),
        }
        return final_tokens, refined_score, meta

    def forward(self, img_embs, cap_embs, cap_lens, long_cap_embs=None, long_cap_lens=None, enable_refine=None):
        B_v, L_v, C = img_embs.shape
        img_embs_norm = F.normalize(img_embs, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)
        long_cap_embs_norm = F.normalize(long_cap_embs, dim=-1) if long_cap_embs is not None else None

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True
        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        with torch.no_grad():
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        if enable_refine is None:
            enable_refine = bool(getattr(self.opt, 'use_reasonable_refine', 0)) and (
                self.training or bool(getattr(self.opt, 'refine_eval', 0))
            )

        cap_text_global_bank = self._build_text_global_bank(cap_embs_norm, cap_lens)
        improve_sims = []
        score_mask_all = []
        necessity_means = []
        exactness_means = []
        pos_margin_means = []
        neg_margin_means = []

        for i in range(len(cap_lens)):
            n_word = int(cap_lens[i])
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)
            cap_i_glo = cap_text_global_bank[i]
            with torch.no_grad():
                attn_cap = (cap_i_glo.unsqueeze(0) * img_spatial_embs_norm).sum(dim=-1)
            select_tokens_cap, extra_token_cap, score_mask_cap, selected_score_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,
                attention_y=attn_cap,
            )
            select_tokens, refined_score, meta = self._compute_refined_tokens(
                img_cls_emb=img_cls_emb,
                selected_tokens=select_tokens_cap,
                extra_token=extra_token_cap,
                selected_score=selected_score_cap,
                pos_text_global=cap_i_glo,
                pos_text_expand=cap_i_expand,
                positive_index=i,
                text_global_bank=cap_text_global_bank,
                text_token_bank=cap_embs_norm,
                text_lens_bank=cap_lens,
                enable_refine=enable_refine,
            )
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens, cap_i_expand=cap_i_expand)
            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask_cap)
            necessity_means.append(meta['necessity_score'])
            exactness_means.append(meta['exactness_score'])
            pos_margin_means.append(meta['refined_global_pos'])
            neg_margin_means.append(meta['refined_global_neg'])

        long_sims = []
        score_mask_long_all = []
        if long_cap_embs is not None and long_cap_lens is not None:
            long_text_global_bank = self._build_text_global_bank(long_cap_embs_norm, long_cap_lens)
            for i in range(len(long_cap_lens)):
                n_word = int(long_cap_lens[i])
                long_cap_i_expand = long_cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)
                long_cap_i_glo = long_text_global_bank[i]
                with torch.no_grad():
                    long_attn_cap = (long_cap_i_glo.unsqueeze(0) * img_spatial_embs_norm).sum(dim=-1)
                select_tokens_long, extra_token_long, score_mask_long, selected_score_long = self.sparse_net_long(
                    tokens=img_spatial_embs,
                    attention_x=img_spatial_self_attention,
                    attention_y=long_attn_cap,
                )
                select_tokens_long, refined_score_long, meta_long = self._compute_refined_tokens(
                    img_cls_emb=img_cls_emb,
                    selected_tokens=select_tokens_long,
                    extra_token=extra_token_long,
                    selected_score=selected_score_long,
                    pos_text_global=long_cap_i_glo,
                    pos_text_expand=long_cap_i_expand,
                    positive_index=i,
                    text_global_bank=long_text_global_bank,
                    text_token_bank=long_cap_embs_norm,
                    text_lens_bank=long_cap_lens,
                    enable_refine=enable_refine,
                )
                sim_one_text = mask_xattn_one_text(img_embs=select_tokens_long, cap_i_expand=long_cap_i_expand)
                long_sims.append(sim_one_text)
                score_mask_long_all.append(score_mask_long)
                necessity_means.append(meta_long['necessity_score'])
                exactness_means.append(meta_long['exactness_score'])
                pos_margin_means.append(meta_long['refined_global_pos'])
                neg_margin_means.append(meta_long['refined_global_neg'])

        improve_sims = torch.cat(improve_sims, dim=1)
        if len(long_sims) > 0:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        score_mask_all_tensor = torch.stack(score_mask_all, dim=0)
        if len(score_mask_long_all) > 0:
            score_mask_all_tensor = score_mask_all_tensor + torch.stack(score_mask_long_all, dim=0)

        aux = {
            'necessity_mean': torch.stack(necessity_means, dim=0).mean(),
            'exactness_mean': torch.stack(exactness_means, dim=0).mean(),
            'pos_margin_mean': torch.stack(pos_margin_means, dim=0).mean(),
            'neg_margin_mean': torch.stack(neg_margin_means, dim=0).mean(),
        }

        if self.training:
            return improve_sims, score_mask_all_tensor, aux
        return improve_sims


if __name__ == '__main__':
    pass
