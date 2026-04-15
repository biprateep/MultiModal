#!/usr/bin/env python3
"""Compare two model checkpoints (e.g. epoch 074 and 113) against the simple
linear ridge-on-DESI-patch baseline from manga_ifu_linear_baseline.ipynb on a
held-out test set of MaNGA-DESI matched galaxies.

Test-set protocol:
- Eligible objects: matches_csv ∩ DESI parquet table.
- 50 random test objects (seed=0) held out for evaluation.
- 80 random training objects (seed=1) drawn from the remaining pool (disjoint
  from test). One shared ridge regressor is trained on all spaxel patches from
  these 80, then reused for every test object. Equivalent to LOO since training
  is fully disjoint from the test set.

Per test object:
- Build linear baseline Halpha map (ridge on 7x7x6 image patches).
- Build pred map for each checkpoint by running the model in batches over all
  inside-DESI valid spaxels and integrating Halpha from predicted spectra.
- Compute metrics on raw and median/IQR-normalized maps.

Aggregate across the test set:
- median + IQR per metric per model,
- win-rate vs the linear baseline per metric,
- paired Wilcoxon p-value (model vs baseline) per metric,
- per-object metrics dropped per request, but a long-form CSV of aggregate
  numbers is written + a summary violin plot.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.eval_halpha_checkpoint_sweep import (
    HA_REST,
    PIX_SCALE,
    PairExtractor,
    build_model,
    extract_halpha_fast_batch,
    parse_epoch,
)


def xy_to_img_pixel(dx, dy, w=128, h=128):
    return dx + w / 2.0, h / 2.0 - dy


def extract_patch_features(img_6ch, x_off, y_off, radius=3):
    C, H, W = img_6ch.shape
    x_img, y_img = xy_to_img_pixel(x_off, y_off, w=W, h=H)
    ix = int(np.round(x_img))
    iy = int(np.round(y_img))
    if ix < 0 or ix >= W or iy < 0 or iy >= H:
        return None
    pad = radius
    img_pad = np.pad(img_6ch, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    patch = img_pad[
        :,
        iy + pad - radius : iy + pad + radius + 1,
        ix + pad - radius : ix + pad + radius + 1,
    ]
    return patch.reshape(-1).astype(np.float64)


def build_target_context(extractor: PairExtractor, desi_tab: pd.DataFrame, targetid: int):
    hit = desi_tab.loc[desi_tab["TARGETID"] == targetid]
    if len(hit) == 0:
        return None
    desi_index = int(hit.index[0])
    desi_row = desi_tab.iloc[desi_index]
    target_ra = float(desi_row["TARGET_RA"])
    target_dec = float(desi_row["TARGET_DEC"])
    z_obj = float(desi_row["Z"])

    plateifu, sep = extractor.match_manga(target_ra, target_dec)
    loader = extractor.get_loader(desi_index)
    batch = next(iter(loader))
    img_6ch = batch[4][0].detach().cpu().numpy().astype(np.float64)

    maps = extractor.get_maps(plateifu)
    truth_ha = np.asarray(maps["ha_flux"], dtype=np.float64)

    maps_path = extractor._maps_path(plateifu)
    with fits.open(maps_path) as h:
        wcs_map = WCS(h["SPX_MFLUX"].header).celestial

    ny, nx = truth_ha.shape
    iy, ix = np.indices((ny, nx))
    ra_grid, dec_grid = wcs_map.pixel_to_world_values(ix.ravel(), iy.ravel())

    x_off = (-1.0 * (ra_grid - target_ra) * np.cos(np.deg2rad(target_dec)) * 3600.0 / PIX_SCALE).reshape(ny, nx)
    y_off = (-1.0 * (dec_grid - target_dec) * 3600.0 / PIX_SCALE).reshape(ny, nx)

    inside_desi = (np.abs(x_off) <= 63) & (np.abs(y_off) <= 63)
    valid = np.isfinite(truth_ha) & inside_desi

    coords = np.column_stack([x_off[valid], y_off[valid]]).astype(np.float32)
    yx_pairs = np.column_stack(np.where(valid)).astype(np.int32)

    return {
        "targetid": targetid,
        "desi_index": desi_index,
        "target_ra": target_ra,
        "target_dec": target_dec,
        "z_obj": z_obj,
        "plateifu": plateifu,
        "sep_arcsec": sep,
        "batch": batch,
        "img_6ch": img_6ch,
        "truth": truth_ha,
        "x_off": x_off,
        "y_off": y_off,
        "coords": coords,
        "yx_pairs": yx_pairs,
    }


def build_training_table(extractor, desi_tab, train_targetids, radius=3):
    X_list, y_list = [], []
    n_used = 0
    for targetid in train_targetids:
        try:
            ctx = build_target_context(extractor, desi_tab, targetid)
        except Exception as e:
            print(f"  skip train tid={targetid}: {e}", flush=True)
            continue
        if ctx is None:
            continue
        truth = ctx["truth"]
        x_off = ctx["x_off"]
        y_off = ctx["y_off"]
        valid = np.isfinite(truth) & ((np.abs(x_off) <= 63) & (np.abs(y_off) <= 63))
        ys, xs = np.where(valid)
        for y, x in zip(ys, xs):
            feat = extract_patch_features(ctx["img_6ch"], x_off[y, x], y_off[y, x], radius=radius)
            if feat is None:
                continue
            X_list.append(feat)
            y_list.append(float(truth[y, x]))
        n_used += 1
        print(f"  train tid={targetid} cum_galaxies={n_used} cum_samples={len(X_list)}", flush=True)
    if not X_list:
        raise RuntimeError("no training samples")
    return np.vstack(X_list), np.asarray(y_list, dtype=np.float64), n_used


def predict_baseline_map(lin_model, ctx, radius=3):
    truth = ctx["truth"]
    x_off = ctx["x_off"]
    y_off = ctx["y_off"]
    img = ctx["img_6ch"]
    out = np.full_like(truth, np.nan, dtype=np.float64)
    valid = np.isfinite(truth) & ((np.abs(x_off) <= 63) & (np.abs(y_off) <= 63))
    ys, xs = np.where(valid)
    feats, locs = [], []
    for y, x in zip(ys, xs):
        f = extract_patch_features(img, x_off[y, x], y_off[y, x], radius=radius)
        if f is None:
            continue
        feats.append(f)
        locs.append((y, x))
    if not feats:
        return out
    pred = lin_model.predict(np.vstack(feats))
    for (y, x), v in zip(locs, pred):
        out[y, x] = float(v)
    return out


def predict_model_map(model, ctx, wave_obs, device, batch_size=1024):
    truth = ctx["truth"]
    coords = ctx["coords"]
    yx_pairs = ctx["yx_pairs"]
    out = np.full_like(truth, np.nan, dtype=np.float64)
    if coords.shape[0] == 0:
        return out

    x, spec, weig, error, img, img_w, img_e, z, xy_pix = ctx["batch"]
    spec = spec.to(device)
    weig = weig.to(device)
    error = error.to(device)
    img = img.to(device)
    img_w = img_w.to(device)
    img_e = img_e.to(device)
    z = z.to(device)
    z_obj = float(ctx["z_obj"])

    n = coords.shape[0]
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bsz = end - start
            xy_chunk = torch.as_tensor(coords[start:end], dtype=xy_pix.dtype, device=device)
            spec_b = spec.expand(bsz, -1)
            weig_b = weig.expand(bsz, -1)
            error_b = error.expand(bsz, -1)
            img_b = img.expand(bsz, -1, -1, -1)
            img_w_b = img_w.expand(bsz, -1, -1, -1)
            img_e_b = img_e.expand(bsz, -1, -1, -1)
            z_b = z.expand(bsz)
            _, _, _, spec_pred, _, _, _, _ = model.forward(
                spec_b, weig_b, error_b, img_b, img_w_b, img_e_b, z_b, xy_chunk
            )
            pred_np = spec_pred.detach().cpu().numpy()
            hvals = extract_halpha_fast_batch(
                pred_np,
                wave_obs=wave_obs,
                z_obj=z_obj,
                left_patches=model.left_patches,
                patch_size=model.patch_size,
            )
            loc = yx_pairs[start:end]
            out[loc[:, 0], loc[:, 1]] = hvals
    return out


def robust_normalize(arr):
    x = np.asarray(arr, dtype=np.float64)
    m = np.isfinite(x)
    if int(m.sum()) < 10:
        return np.full_like(x, np.nan)
    v = x[m]
    med = float(np.nanmedian(v))
    q25, q75 = np.nanpercentile(v, [25, 75])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 0:
        s = float(np.nanstd(v))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        return (x - med) / s
    return (x - med) / iqr


def compute_metrics(pred, truth):
    m = np.isfinite(pred) & np.isfinite(truth)
    n = int(m.sum())
    out = {
        "n_pix": n,
        "pearson": np.nan,
        "spearman": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "rmse_norm": np.nan,
        "mae_norm": np.nan,
        "cosine_norm": np.nan,
        "r2_norm": np.nan,
    }
    if n < 10:
        return out
    p = pred[m].astype(np.float64)
    t = truth[m].astype(np.float64)
    out["pearson"] = float(np.corrcoef(p, t)[0, 1])
    try:
        rho, _ = spearmanr(p, t)
        out["spearman"] = float(rho)
    except Exception:
        pass
    out["rmse"] = float(np.sqrt(np.mean((p - t) ** 2)))
    out["mae"] = float(np.mean(np.abs(p - t)))
    out["bias"] = float(np.mean(p - t))

    p_n = robust_normalize(p)
    t_n = robust_normalize(t)
    out["rmse_norm"] = float(np.sqrt(np.mean((p_n - t_n) ** 2)))
    out["mae_norm"] = float(np.mean(np.abs(p_n - t_n)))
    denom = np.linalg.norm(p_n) * np.linalg.norm(t_n)
    out["cosine_norm"] = float(np.dot(p_n, t_n) / denom) if denom > 0 else np.nan
    ss_res = float(np.sum((t_n - p_n) ** 2))
    ss_tot = float(np.sum((t_n - np.mean(t_n)) ** 2))
    out["r2_norm"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return out


HIGHER_BETTER = {"pearson", "spearman", "cosine_norm", "r2_norm"}
LOWER_BETTER = {"rmse", "mae", "rmse_norm", "mae_norm"}
SIGNED = {"bias"}


def aggregate(rows: pd.DataFrame, baseline_name: str):
    metrics = [
        "pearson",
        "spearman",
        "cosine_norm",
        "r2_norm",
        "rmse",
        "mae",
        "rmse_norm",
        "mae_norm",
        "bias",
        "n_pix",
    ]
    methods = list(rows["method"].unique())
    summary = []
    for method in methods:
        sub = rows[rows["method"] == method]
        for met in metrics:
            v = pd.to_numeric(sub[met], errors="coerce").dropna().values
            if len(v) == 0:
                continue
            row = {
                "method": method,
                "metric": met,
                "n_obj": int(len(v)),
                "median": float(np.median(v)),
                "iqr_lo": float(np.percentile(v, 25)),
                "iqr_hi": float(np.percentile(v, 75)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            }
            summary.append(row)
    summary_df = pd.DataFrame(summary)

    pair_rows = []
    base = rows[rows["method"] == baseline_name].set_index("targetid")
    for method in methods:
        if method == baseline_name:
            continue
        cur = rows[rows["method"] == method].set_index("targetid")
        joined = base.join(cur, lsuffix="_base", rsuffix="_mod", how="inner")
        for met in metrics:
            if met == "n_pix":
                continue
            b = pd.to_numeric(joined[f"{met}_base"], errors="coerce")
            m = pd.to_numeric(joined[f"{met}_mod"], errors="coerce")
            mask = b.notna() & m.notna()
            if mask.sum() < 5:
                continue
            b = b[mask].values
            m = m[mask].values
            if met in HIGHER_BETTER:
                wins = int(np.sum(m > b))
            elif met in LOWER_BETTER:
                wins = int(np.sum(m < b))
            else:
                wins = int(np.sum(np.abs(m) < np.abs(b)))
            try:
                stat, p = wilcoxon(m - b, zero_method="wilcox", alternative="two-sided")
                p_val = float(p)
            except Exception:
                p_val = np.nan
            pair_rows.append({
                "method": method,
                "vs": baseline_name,
                "metric": met,
                "n_pairs": int(mask.sum()),
                "wins": wins,
                "win_rate": float(wins / mask.sum()),
                "median_diff": float(np.median(m - b)),
                "wilcoxon_p": p_val,
            })
    pair_df = pd.DataFrame(pair_rows)
    return summary_df, pair_df


def violin_plot(rows: pd.DataFrame, out_png: Path):
    metrics = ["cosine_norm", "pearson", "spearman", "rmse_norm", "mae_norm", "r2_norm"]
    methods = list(rows["method"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, met in zip(axes.flat, metrics):
        data = []
        labels = []
        for method in methods:
            v = pd.to_numeric(rows[rows["method"] == method][met], errors="coerce").dropna().values
            if len(v):
                data.append(v)
                labels.append(method)
        if not data:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(met)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-object metric distributions over test set")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-a", default="/pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsSimple/epoch=074-val_loss=-1.5135.ckpt")
    ap.add_argument("--checkpoint-b", default="/pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsSimple/epoch=113-val_loss=-1.5626.ckpt")
    ap.add_argument("--matches-csv", default="/pscratch/sd/p/pzehao/desi_manga_matches.csv")
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--n-train", type=int, default=0)  # 0 = use all remaining (eligible - test)
    ap.add_argument("--patch-radius", type=int, default=3)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ridge-cache", type=str, default="")
    ap.add_argument("--y-clip-pct", type=float, default=99.9,
                    help="drop training spaxels with y above this percentile "
                         "(MaNGA DAP maps contain rare extreme outliers that poison Ridge)")
    ap.add_argument("--seed-test", type=int, default=0)
    ap.add_argument("--seed-train", type=int, default=1)
    ap.add_argument("--output-dir", default="/pscratch/sd/p/pzehao/MultiModal/notebooks/two_ckpts_vs_baseline")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = PairExtractor()
    desi_tab = extractor.desi_table()
    matches = pd.read_csv(args.matches_csv)
    elig = matches[matches["TARGETID"].isin(set(desi_tab["TARGETID"].values))].reset_index(drop=True)
    uniq = pd.unique(elig["TARGETID"].astype(np.int64))
    print(f"eligible unique targetids: {len(uniq)}", flush=True)

    rng_t = np.random.default_rng(args.seed_test)
    rng_tr = np.random.default_rng(args.seed_train)
    test_ids = list(rng_t.choice(uniq, size=min(args.n_test, len(uniq)), replace=False))
    pool = [int(t) for t in uniq if int(t) not in set(test_ids)]
    if args.n_train and args.n_train > 0:
        train_ids = list(rng_tr.choice(pool, size=min(args.n_train, len(pool)), replace=False))
    else:
        train_ids = list(pool)
    test_ids = [int(t) for t in test_ids]
    train_ids = [int(t) for t in train_ids]

    print(f"n_test={len(test_ids)} n_train={len(train_ids)}", flush=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"device={device}", flush=True)

    ridge_cache = Path(args.ridge_cache) if args.ridge_cache else (out_dir / "ridge_model.joblib")
    if ridge_cache.exists():
        print(f"== loading cached ridge from {ridge_cache} ==", flush=True)
        bundle = joblib.load(ridge_cache)
        lin_model = bundle["model"]
        n_train_used = int(bundle.get("n_train_used", -1))
    else:
        print("== building training table ==", flush=True)
        t0 = time.time()
        X_tr, y_tr, n_train_used = build_training_table(extractor, desi_tab, train_ids, radius=args.patch_radius)
        print(f"train X={X_tr.shape} y={y_tr.shape} from {n_train_used} galaxies in {time.time()-t0:.1f}s", flush=True)

        if args.y_clip_pct and args.y_clip_pct < 100:
            cap = float(np.percentile(y_tr, args.y_clip_pct))
            keep = y_tr <= cap
            print(f"y cap@p{args.y_clip_pct}={cap:.3f}; dropping {(~keep).sum()}/{len(y_tr)} spaxels", flush=True)
            X_tr, y_tr = X_tr[keep], y_tr[keep]

        print("== fitting ridge ==", flush=True)
        lin_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.ridge_alpha, random_state=0)),
        ])
        lin_model.fit(X_tr, y_tr)
        joblib.dump({"model": lin_model, "n_train_used": n_train_used}, ridge_cache)
        print(f"saved ridge to {ridge_cache}", flush=True)

    print("== building models ==", flush=True)
    model_a = build_model().to(device)
    chk_a = torch.load(args.checkpoint_a, map_location="cpu")
    model_a.load_state_dict(chk_a["state_dict"], strict=False)
    model_a.eval()

    model_b = build_model().to(device)
    chk_b = torch.load(args.checkpoint_b, map_location="cpu")
    model_b.load_state_dict(chk_b["state_dict"], strict=False)
    model_b.eval()

    label_a = f"ckpt_epoch_{parse_epoch(args.checkpoint_a):03d}"
    label_b = f"ckpt_epoch_{parse_epoch(args.checkpoint_b):03d}"

    wave_obs = np.asarray(extractor.wave, dtype=np.float64)

    rows = []
    print("== iterating test objects ==", flush=True)
    for i, tid in enumerate(test_ids, start=1):
        ck_t = time.time()
        try:
            ctx = build_target_context(extractor, desi_tab, tid)
        except Exception as e:
            print(f"[{i}/{len(test_ids)}] tid={tid} ctx_fail: {e}", flush=True)
            continue
        if ctx is None:
            continue
        truth = ctx["truth"]

        base_map = predict_baseline_map(lin_model, ctx, radius=args.patch_radius)
        m_base = compute_metrics(base_map, truth)
        m_base["method"] = "linear_baseline"

        a_map = predict_model_map(model_a, ctx, wave_obs, device, batch_size=args.batch_size)
        m_a = compute_metrics(a_map, truth)
        m_a["method"] = label_a
        if device.type == "cuda":
            torch.cuda.empty_cache()

        b_map = predict_model_map(model_b, ctx, wave_obs, device, batch_size=args.batch_size)
        m_b = compute_metrics(b_map, truth)
        m_b["method"] = label_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for d in (m_base, m_a, m_b):
            d["targetid"] = tid
            d["plateifu"] = ctx["plateifu"]
            rows.append(d)

        elapsed = time.time() - ck_t
        print(
            f"[{i}/{len(test_ids)}] tid={tid} pifu={ctx['plateifu']} npix={m_base['n_pix']} "
            f"cos_base={m_base['cosine_norm']:.4f} cos_a={m_a['cosine_norm']:.4f} "
            f"cos_b={m_b['cosine_norm']:.4f} sec={elapsed:.1f}",
            flush=True,
        )

    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_dir / "perobj_metrics.csv", index=False)

    summary_df, pair_df = aggregate(long_df, baseline_name="linear_baseline")
    summary_df.to_csv(out_dir / "aggregate_summary.csv", index=False)
    pair_df.to_csv(out_dir / "paired_vs_baseline.csv", index=False)

    violin_plot(long_df, out_dir / "metric_violins.png")

    meta = {
        "n_test_attempted": len(test_ids),
        "n_test_completed": int(long_df["targetid"].nunique()) if len(long_df) else 0,
        "n_train_galaxies_used": int(n_train_used),
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "label_a": label_a,
        "label_b": label_b,
        "patch_radius": args.patch_radius,
        "ridge_alpha": args.ridge_alpha,
        "seed_test": args.seed_test,
        "seed_train": args.seed_train,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== SUMMARY (median [IQR]) ===", flush=True)
    if len(summary_df):
        for method in summary_df["method"].unique():
            print(f"\n[{method}]", flush=True)
            sub = summary_df[summary_df["method"] == method]
            for _, r in sub.iterrows():
                print(
                    f"  {r['metric']:>12s}: median={r['median']:.4f} IQR=[{r['iqr_lo']:.4f},{r['iqr_hi']:.4f}] "
                    f"mean={r['mean']:.4f} std={r['std']:.4f} n_obj={int(r['n_obj'])}",
                    flush=True,
                )

    print("\n=== PAIRED vs linear_baseline ===", flush=True)
    if len(pair_df):
        for method in pair_df["method"].unique():
            print(f"\n[{method} vs linear_baseline]", flush=True)
            sub = pair_df[pair_df["method"] == method]
            for _, r in sub.iterrows():
                print(
                    f"  {r['metric']:>12s}: win_rate={r['win_rate']:.2f} ({int(r['wins'])}/{int(r['n_pairs'])})  "
                    f"median_diff={r['median_diff']:+.4f}  wilcoxon_p={r['wilcoxon_p']:.3g}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
