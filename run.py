"""
run.py — Standard entry point for Lyra (Thuml convention).

Usage:
    python run.py --data ETTh1 --seq_len 336 --pred_len 96 --features M
"""

import os
import argparse
import random
import logging
import numpy as np
import torch

from exp.exp_main import Exp_Main

# ──────────────────────────────────────────────────────────────────────
# Global logging — exp_main.py uses log.info() throughout
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Lyra — Time Series Forecasting")

    # ── Data ──────────────────────────────────────────────────────────
    parser.add_argument("--data", type=str, default="ETTh1", help="Dataset name")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M",
                        help="M: multivariate->multivariate, S: univariate->univariate, MS: multi->uni")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # ── Forecasting ───────────────────────────────────────────────────
    parser.add_argument("--seq_len", type=int, default=336, help="Lookback window")
    parser.add_argument("--label_len", type=int, default=48, help="Label length (decoder)")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecast horizon")

    # ── Model (Lyra specific) ─────────────────────────────────────────
    parser.add_argument("--enc_in", type=int, default=7, help="Number of input channels")
    parser.add_argument("--d_model", type=int, default=512, help="GLU-MLP hidden dimension")
    parser.add_argument("--cond_threshold", type=float, default=1e6,
                        help="Condition number threshold for Tikhonov regularization")
    parser.add_argument("--n_power_iters", type=int, default=3,
                        help="Power iteration steps for spectral norm estimation")
    parser.add_argument("--ridge_lambda", type=float, default=5.0,
                        help="Ridge damping for Vandermonde trend extrapolation")
    parser.add_argument("--dropout", type=float, default=0.45,
                        help="Dropout rate for GLU-MLP")

    # ── Training ──────────────────────────────────────────────────────
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0, help="Data loader num workers")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Default 3e-4; 1e-3 may converge faster for larger d_model")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="AdamW weight decay (regularization)")
    parser.add_argument("--lradj", type=str, default="none")
    parser.add_argument("--embed", type=str, default="fixed")

    # ── Cache ─────────────────────────────────────────────────────────
    # choices use ASCII arrows to match the values accepted by exp_main.py
    parser.add_argument("--cache_mode", type=str, default="auto",
                        choices=["gpu", "ram", "disk->ram", "disk->stream", "auto"],
                        help="Cache routing: gpu | ram | disk->ram | disk->stream | auto")
    parser.add_argument("--gpu_output_ratio", type=float, default=0.85,
                        help="AutoCache GPU output threshold ratio (of usable VRAM)")
    parser.add_argument("--gpu_compute_ratio", type=float, default=0.75,
                        help="AutoCache GPU compute peak threshold ratio (of usable VRAM)")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Delete existing cache before training "
                             "(use after changing seq_len / pred_len / enc_in)")
    parser.add_argument("--train_mode", type=str, default="cached",
                        choices=["cached", "e2e"],
                        help="Training mode: cached=Phase1+Phase2, e2e=end-to-end")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (pass different values for multi-seed evaluation)")

    # ── GPU ───────────────────────────────────────────────────────────
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_shuffle", type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        default=True, help="1 to enable GPU internal shuffle, 0 to disable")


    # ── Mode ──────────────────────────────────────────────────────────
    parser.add_argument("--is_training", type=int, default=1, help="1=train, 0=test only")

    args = parser.parse_args()

    # ── Reproducibility — applied here so --seed takes effect ─────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True   # exact reproducibility over minor conv speed
        torch.backends.cudnn.benchmark = False

    log.info("=" * 60)
    log.info("  Lyra time-series forecasting")
    log.info("  Cached training pipeline: Cache -> Train -> Test")
    log.info("=" * 60)
    log.info(f"  Data: {args.data} | Lookback: {args.seq_len} | Horizon: {args.pred_len}")
    log.info(f"  Channels: {args.enc_in} | Hidden: {args.d_model}")
    log.info(f"  Power Iters: {args.n_power_iters}")
    log.info(f"  Cache Mode: {args.cache_mode} | Clear Cache: {args.clear_cache}")
    log.info(f"  Train Mode: {args.train_mode} | Seed: {args.seed}")
    log.info("=" * 60)

    setting = (
        f"Lyra_{args.data}_sl{args.seq_len}_pl{args.pred_len}"
        f"_dm{args.d_model}"
    )

    exp = Exp_Main(args)

    if args.is_training:
        log.info(f">>> Training: {setting}")
        exp.train(setting)

        log.info(f">>> Testing: {setting}")
        exp.test(setting, load_checkpoint=True)
    else:
        log.info(f">>> Testing only: {setting}")
        exp.test(setting, load_checkpoint=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
