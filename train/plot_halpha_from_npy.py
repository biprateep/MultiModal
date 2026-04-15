#!/usr/bin/env python3
"""Render Halpha map PNGs from .npy files saved by eval_halpha_checkpoint_sweep.py.

Loads truth context (same target row) once, then for every halpha_map_*.npy in
--input-dir produces a 3-panel (truth / pred / diff) PNG alongside.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.eval_halpha_checkpoint_sweep import (
    PairExtractor,
    cosine_norm,
    load_target_context,
    parse_epoch,
    plot_truth_pred_diff,
)


def main():
    ap = argparse.ArgumentParser(description="Plot Halpha maps from saved .npy files")
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--matches-csv", type=str, default="/pscratch/sd/p/pzehao/desi_manga_matches.csv")
    ap.add_argument("--target-row-idx", type=int, default=2)
    ap.add_argument("--metrics-csv", type=str, default="")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    npy_files = sorted(in_dir.glob("halpha_map_*.npy"))
    if not npy_files:
        raise RuntimeError(f"No halpha_map_*.npy found in {in_dir}")

    extractor = PairExtractor()
    ctx = load_target_context(extractor, args.matches_csv, args.target_row_idx)
    truth = ctx["truth"]

    cos_lookup = {}
    if args.metrics_csv and Path(args.metrics_csv).exists():
        df = pd.read_csv(args.metrics_csv)
        for _, row in df.iterrows():
            cos_lookup[str(row["checkpoint_name"])] = float(row["cosine_norm"])

    for npy in npy_files:
        pred = np.load(npy).astype(np.float64)
        ckpt_name = npy.name.replace("halpha_map_", "").replace(".npy", ".ckpt")
        ep = parse_epoch(ckpt_name)
        cos = cos_lookup.get(ckpt_name)
        if cos is None:
            cos = cosine_norm(pred, truth)
        title = (
            f"targetid={ctx['targetid']} plateifu={ctx['plateifu']} "
            f"epoch={ep} cos={cos:.4f}"
        )
        out_png = npy.with_suffix(".png")
        plot_truth_pred_diff(truth, pred, out_png, title)
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
