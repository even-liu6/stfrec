import argparse


DIFFUSION_DEFAULTS = {
    "douban-book": {"rw_alpha": 0.70, "rw_beta": 0.30, "rw_topk": 16},
    "tmall": {"rw_alpha": 0.40, "rw_beta": 0.30, "rw_topk": 16},
    "amazon-book": {"rw_alpha": 0.40, "rw_beta": 0.70, "rw_topk": 16},
}


def str2bool(v):
    if isinstance(v, bool):
        return v

    v = v.lower()

    if v in ("yes", "true", "t", "1"):
        return True

    if v in ("no", "false", "f", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(argv=None):
    parse = argparse.ArgumentParser(description="Run stfrec")

    # ===== Basic =====
    parse.add_argument("--seed", type=int, default=2023, help="random seed")
    parse.add_argument("--gpu", type=int, default=0, help="indicates which gpu to use")
    parse.add_argument("--cuda", type=str2bool, default=True, help="use gpu or not")

    parse.add_argument("--log", type=str, default="None", help="init log file name")
    parse.add_argument("--dataset_path", type=str, default="./dataset/", help="dataset path")
    parse.add_argument("--dataset_type", type=str, default=".txt", help="dataset file type")
    parse.add_argument("--dataset", type=str, default="douban-book", help="dataset name")

    parse.add_argument("--top_K", type=str, default="[10, 20, 30, 40, 50]")
    parse.add_argument("--train_epoch", type=int, default=600)
    parse.add_argument("--early_stop", type=int, default=10)
    parse.add_argument("--embedding_size", type=int, default=64)
    parse.add_argument("--train_batch_size", type=int, default=2048)
    parse.add_argument("--test_batch_size", type=int, default=2048)
    parse.add_argument("--learn_rate", type=float, default=0.001)
    parse.add_argument("--reg_lambda", type=float, default=0.0001)
    parse.add_argument("--gcn_layer", type=int, default=3)
    parse.add_argument("--test_frequency", type=int, default=1)
    parse.add_argument("--sparsity_test", type=int, default=0)

    # ===== Inherited LightCCF parameters =====
    parse.add_argument("--tau", type=float, default=0.28)
    parse.add_argument("--ssl_lambda", type=float, default=1.0)
    parse.add_argument("--encoder", type=str, default="MF")

    # ===== Recognized old experimental switches, default OFF =====
    parse.add_argument("--debias_na", type=int, default=0)
    parse.add_argument("--pop_gamma", type=float, default=0.5)
    parse.add_argument("--pop_w_min", type=float, default=0.5)
    parse.add_argument("--pop_w_max", type=float, default=2.0)
    parse.add_argument("--gb_mix", type=float, default=0.0)
    parse.add_argument("--tail_ratio", type=float, default=0.5)
    parse.add_argument("--middle_ratio", type=float, default=0.3)
    parse.add_argument("--pop_bpr", type=int, default=0)
    parse.add_argument("--pop_eval", type=int, default=0)

    parse.add_argument(
        "--na_loss_type",
        type=str,
        default="origin",
        choices=["origin", "fnm", "ucna", "hybrid"],
        help="NA loss type"
    )
    parse.add_argument("--ucna_lambda", type=float, default=0.5)

    # This parser keeps these names only to avoid command-line errors.
    # The clean trainer below does not enable semi-hard sampling.
    parse.add_argument(
        "--neg_sample",
        type=str,
        default="uniform",
        choices=["uniform", "semi_hard_band", "semi_hard_aux", "soft_hard"],
        help="negative sampling strategy"
    )
    parse.add_argument("--neg_candidates", type=int, default=64)
    parse.add_argument("--neg_warmup", type=int, default=8)
    parse.add_argument("--neg_anneal_epochs", type=int, default=30)
    parse.add_argument("--neg_mix_prob", type=float, default=0.5)
    parse.add_argument("--hard_low_start", type=float, default=0.35)
    parse.add_argument("--hard_high_start", type=float, default=0.70)
    parse.add_argument("--hard_low_end", type=float, default=0.10)
    parse.add_argument("--hard_high_end", type=float, default=0.40)
    parse.add_argument("--hard_bpr_weight", type=float, default=0.0)

    parse.add_argument("--hist_alpha", type=float, default=0.0)

    # ===== Structural diffusion fusion =====
    parse.add_argument("--rw_alpha", type=float, default=None,
                       help="diffusion weight: 0.70 for douban-book, 0.40 for tmall/amazon-book; 0 disables diffusion")
    parse.add_argument("--rw_topk", type=int, default=None,
                       help="neighbors retained per item in the diffusion graph (default: 16)")
    parse.add_argument("--rw_beta", type=float, default=None,
                       help="popularity penalty: 0.70 for amazon-book, 0.30 for douban-book/tmall")
    parse.add_argument("--rw_candidate_k", type=int, default=300)

    # Disable decay by default.
    parse.add_argument("--rw_decay_start", type=int, default=1000000)
    parse.add_argument("--rw_decay_rate", type=float, default=1.0)
    parse.add_argument("--rw_min_ratio", type=float, default=1.0)


    # ===== Ablation switches for SDF =====
    parse.add_argument("--rw_user_norm", type=int, default=1,
                       help="1: use D_u^-1 in R^T D_u^-1 R; 0: use R^T R")

    parse.add_argument("--rw_pop_penalty", type=int, default=1,
                       help="1: use target-item popularity penalty; 0: disable it")

    parse.add_argument("--rw_score_norm", type=str, default="minmax",
                       choices=["minmax", "none"],
                       help="minmax: user-wise min-max normalization; none: raw diffusion score")

    args = parse.parse_args(argv)
    # Custom datasets use the Douban-book preset unless explicitly overridden.
    defaults = DIFFUSION_DEFAULTS.get(args.dataset, DIFFUSION_DEFAULTS["douban-book"])
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args
