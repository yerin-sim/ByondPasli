import argparse
import os


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='data/', type=str, help='path to datasets')
    parser.add_argument('--dataset', default='f30k', help='dataset coco or f30k')

    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=512, type=int, help='Dimensionality of the joint embedding.')

    parser.add_argument('--grad_clip', default=2.0, type=float, help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate.')

    parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int, help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int, help='Number of steps to run validation.')

    parser.add_argument('--logger_name', default='runs/test', help='Path to save Tensorboard log.')

    # staged refinement / resume
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--load_optimizer', default=0, type=int, help='load optimizer state when resuming')
    parser.add_argument('--reasonable_start_epoch', default=9, type=int, help='epoch to start enabling second-stage refinement')
    parser.add_argument('--reasonable_warmup_epochs', default=6, type=int, help='epochs to linearly warm up refinement losses')
    parser.add_argument('--use_reasonable_refine', default=1, type=int, help='enable necessity/exactness refinement')
    parser.add_argument('--refine_eval', default=0, type=int, help='enable refinement during evaluation')
    parser.add_argument('--refine_ratio', default=0.85, type=float, help='ratio of sparse-selected patches kept before aggregation')

    # auxiliary refinement losses
    parser.add_argument('--cf_loss_weight', default=0.02, type=float, help='weight for necessity refinement loss')
    parser.add_argument('--exact_loss_weight', default=0.05, type=float, help='weight for exactness refinement loss')
    parser.add_argument('--necessity_margin', default=0.00, type=float, help='target lower bound for mean necessity score')
    parser.add_argument('--exact_margin', default=0.05, type=float, help='target lower bound for mean exactness score')
    parser.add_argument('--patch_refine_start_epoch', default=3, type=int)

    # fused pre-aggregation reranking weights
    parser.add_argument('--score_attn_weight', default=0.15, type=float, help='weak prior weight for sparse attention score')
    parser.add_argument('--score_nec_weight', default=0.45, type=float, help='weight for ownership-based necessity score')
    parser.add_argument('--score_exact_weight', default=0.30, type=float, help='weight for negative-aware exactness score')
    parser.add_argument('--score_red_weight', default=0.10, type=float, help='penalty weight for redundancy score')

    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1, help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=0, help='The number of epochs for warming up the embedding layer')

    parser.add_argument('--f30k_img_path', type=str, default="/home/ssim0425/Patch/SEPS/data/f30k-images", help='the path of f30k images')
    parser.add_argument('--coco_img_path', type=str, default="/home/ssim0425/Patch/SEPS/data/coco-images", help='the path of coco images')

    # vision transformer
    parser.add_argument('--img_res', type=int, default=224, help='the image resolution for ViT input')
    parser.add_argument('--vit_type', type=str, default='vit', help='the type of vit model')

    # distributed training
    parser.add_argument('--multi_gpu', type=int, default=0, help='whether use multi-gpu for training')
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument("--rank", type=int, default=0, help='the parameter for rank')
    parser.add_argument("--local_rank", type=int, default=0, help='the parameter for local rank')

    parser.add_argument('--dist_backend', type=str, default='nccl', help='the backend for ddp')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--seed', type=int, default=0, help='fix the seed for reproducibility')

    # others
    parser.add_argument('--size_augment', type=int, default=1, help='whether use the Size Augmentation')
    parser.add_argument('--loss', type=str, default='vse', help='the objective function for optimization')
    parser.add_argument('--eval', type=int, default=1, help='whether evaluation after training process')

    parser.add_argument('--save_results', type=int, default=1, help='whether save the evaluation results')
    parser.add_argument('--evaluate_cxc', type=int, default=0, help='the special evaluation for MS-COCO')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for running')

    parser.add_argument('--bert_path', type=str, default='../weights_models/bert-base-uncased')

    # optimizer
    parser.add_argument("--lr_schedules", default=[9, 15, 20, 25], type=int, nargs="+", help='epoch schedules for lr decay')
    parser.add_argument("--decay_rate", default=0.3, type=float, help='lr decay_rate for optimizer')

    parser.add_argument('--shard_size', type=int, default=256, help='the shard_size for cross-attention')
    parser.add_argument('--max_word', type=int, default=90, help='the max length for word features')

    # cross-modal alignment
    parser.add_argument('--aggr_ratio', type=float, default=0.4, help='the aggr rate for visual token')
    parser.add_argument('--sparse_ratio', type=float, default=0.5, help='the sparse rate for visual token')
    parser.add_argument('--attention_weight', type=float, default=0.8, help='the weight of attention_map for mask prediction')
    parser.add_argument('--ratio_weight', type=float, default=2.0, help='weight for sparse-ratio regularization')
    parser.add_argument('--use_ratio_loss', type=int, default=1, help='whether to use sparse ratio regularization')

    return parser


def save_parameters(opt, save_path):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t' + str(sub_key) + ': ' + str(sub_item)
        else:
            base_str += '\n\t' + str(varx[key])
        base_str += '\n'

    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


if __name__ == '__main__':
    pass
