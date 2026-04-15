#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.MAESimple import MaskedAutoencoderViT
from utils.DataProcessing import MultimodalDataset

HA_REST = 6564.61
PIX_SCALE = 0.262


class PairExtractor:
    def __init__(
        self,
        desi_zarr_path="/pscratch/sd/p/pzehao/iron/desi_maglim_19_5.zarr",
        desi_parquet_path="/pscratch/sd/p/pzehao/iron/desi_zcat_maglim_19_5.parquet",
        drpall_cache="/pscratch/sd/p/pzehao/MyQuota/drpall-v3_1_1.fits",
        maps_cache_dir="/pscratch/sd/p/pzehao/MyQuota/manga_maps",
        map_daptype="HYB10-MILESHC-MASTARHC2",
    ):
        self.desi_zarr_path = Path(desi_zarr_path)
        self.desi_parquet_path = Path(desi_parquet_path)
        self.drpall_cache = Path(drpall_cache)
        self.maps_cache_dir = Path(maps_cache_dir)
        self.map_daptype = map_daptype

        self.z = zarr.open(str(self.desi_zarr_path), mode="r")
        self.wave = np.asarray(self.z["WAVE"][:], dtype=np.float32)
        self._desi = None
        self._manga = None

    @staticmethod
    def _txt(x):
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8")
        return str(x)

    def desi_table(self):
        if self._desi is None:
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "MEAN_FIBER_RA",
                "MEAN_FIBER_DEC",
                "Z",
            ]
            t = pd.read_parquet(self.desi_parquet_path, columns=cols).copy()
            t["TARGETID"] = t["TARGETID"].astype(np.int64)
            self._desi = t
        return self._desi

    def manga_table(self):
        if self._manga is None:
            if not self.drpall_cache.exists():
                self.drpall_cache.parent.mkdir(parents=True, exist_ok=True)
                url = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"
                urlretrieve(url, self.drpall_cache)
            tbl = Table.read(self.drpall_cache, hdu=1)
            one_d = [c for c in tbl.colnames if len(tbl[c].shape) <= 1]
            df = tbl[one_d].to_pandas()
            df = df[["plateifu", "objra", "objdec"]].copy()
            df["plateifu"] = df["plateifu"].map(self._txt)
            df = df.rename(columns={"objra": "manga_ra", "objdec": "manga_dec"})
            self._manga = df
        return self._manga

    def get_loader(self, desi_index):
        ds = MultimodalDataset(
            str(self.desi_zarr_path),
            start=int(desi_index),
            end=int(desi_index) + 1,
            augment=False,
            max_shift=0,
        )
        return DataLoader(ds, batch_size=1, shuffle=False)

    def match_manga(self, target_ra, target_dec):
        mt = self.manga_table()
        c1 = SkyCoord([target_ra] * u.deg, [target_dec] * u.deg)
        c2 = SkyCoord(mt["manga_ra"].to_numpy() * u.deg, mt["manga_dec"].to_numpy() * u.deg)
        idx, sep2d, _ = c1.match_to_catalog_sky(c2)
        return mt.iloc[int(idx[0])]["plateifu"], float(sep2d.arcsec[0])

    def _maps_path(self, plateifu):
        plate, ifu = plateifu.split("-")
        self.maps_cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.maps_cache_dir / f"manga-{plateifu}-MAPS-{self.map_daptype}.fits.gz"
        if not p.exists():
            u = (
                "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/"
                f"{self.map_daptype}/{plate}/{ifu}/manga-{plateifu}-MAPS-{self.map_daptype}.fits.gz"
            )
            urlretrieve(u, p)
        return p

    @staticmethod
    def _ch(header, token):
        token = token.lower()
        for k, v in header.items():
            if isinstance(k, str) and k.startswith("C") and isinstance(v, str) and token in v.lower():
                return int(k[1:]) - 1
        raise KeyError(token)

    @staticmethod
    def _masked(h, ext, idx=None):
        d = np.asarray(h[ext].data if idx is None else h[ext].data[idx], dtype=np.float32)
        bad = np.zeros_like(d, dtype=bool)
        me = f"{ext}_MASK"
        ie = f"{ext}_IVAR"
        if me in h:
            m = np.asarray(h[me].data if idx is None else h[me].data[idx])
            bad |= m != 0
        if ie in h:
            iv = np.asarray(h[ie].data if idx is None else h[ie].data[idx], dtype=np.float32)
            bad |= (~np.isfinite(iv)) | (iv <= 0)
        d[bad] = np.nan
        return d

    def get_maps(self, plateifu):
        p = self._maps_path(plateifu)
        with fits.open(p) as h:
            i_ha = self._ch(h["EMLINE_GFLUX"].header, "Ha-6564")
            return {
                "manga_image": self._masked(h, "SPX_MFLUX"),
                "ha_flux": self._masked(h, "EMLINE_GFLUX", i_ha),
            }


def _robust_normalize_map(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).copy()
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


def plot_truth_pred_diff(truth: np.ndarray, pred: np.ndarray, out_png: Path, title: str):
    valid = np.isfinite(truth) & np.isfinite(pred)
    if int(valid.sum()) < 10:
        return

    truth_n = _robust_normalize_map(truth)
    pred_n = _robust_normalize_map(pred)
    diff_n = pred_n - truth_n

    both = np.concatenate([truth_n[valid], pred_n[valid]])
    v = float(np.nanpercentile(np.abs(both), 98))
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    d = float(np.nanpercentile(np.abs(diff_n[valid]), 98))
    if not np.isfinite(d) or d <= 0:
        d = 1.0

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = ax[0].imshow(truth_n, origin="lower", cmap="coolwarm", vmin=-v, vmax=v)
    ax[0].set_title("Truth Halpha (normalized)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(pred_n, origin="lower", cmap="coolwarm", vmin=-v, vmax=v)
    ax[1].set_title("Predicted Halpha (normalized)")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(diff_n, origin="lower", cmap="RdBu_r", vmin=-d, vmax=d)
    ax[2].set_title("Pred - Truth (normalized)")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_epoch(ckpt_path: str):
    m = re.search(r"epoch=(\d+)", ckpt_path)
    return int(m.group(1)) if m else -1


def robust_normalize_1d(arr: np.ndarray):
    x = np.asarray(arr, dtype=np.float64)
    m = np.isfinite(x)
    if int(m.sum()) < 10:
        return np.full_like(x, np.nan, dtype=np.float64)
    v = x[m]
    med = np.nanmedian(v)
    q25, q75 = np.nanpercentile(v, [25, 75])
    iqr = q75 - q25
    if (not np.isfinite(iqr)) or (iqr <= 0):
        s = np.nanstd(v)
        if (not np.isfinite(s)) or (s <= 0):
            s = 1.0
        return (x - med) / s
    return (x - med) / iqr


def cosine_norm(pred_map: np.ndarray, truth_map: np.ndarray):
    m = np.isfinite(pred_map) & np.isfinite(truth_map)
    n = int(m.sum())
    if n < 10:
        return np.nan
    p = robust_normalize_1d(pred_map[m])
    t = robust_normalize_1d(truth_map[m])
    denom = np.linalg.norm(p) * np.linalg.norm(t)
    if denom <= 0:
        return np.nan
    return float(np.dot(p, t) / denom)


def build_model():
    prob = 0.7 / 15
    model = MaskedAutoencoderViT(
        spec_dim=7781,
        max_epochs=600,
        warmup_epoch=5,
        mask_ratio=0.75,
        embed_dim=256,
        merged_depth=4,
        merged_num_heads=8,
        s_depth=4,
        e_depth=4,
        s_num_heads=8,
        e_num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_MLP_coefficient=1,
        patch_scheme={
            "patch_sizes": [1, 1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1],
            "mask_ratios": [1, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0],
            "probs": [0.3, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob],
        },
    )
    model.mask_ratio = 1.0
    model.mask_ratio_img = 0.0
    model.eval()
    return model


def extract_halpha_fast_batch(spectra: np.ndarray, wave_obs: np.ndarray, z_obj: float, left_patches: int, patch_size: int):
    arr = np.asarray(spectra, dtype=np.float64)
    L = len(wave_obs)
    if arr.shape[1] != L:
        off = int(left_patches) * int(patch_size)
        arr = arr[:, off : off + L]

    lam0 = HA_REST * (1.0 + z_obj)
    win_half = 45.0 * (1.0 + z_obj)
    m_win = (wave_obs >= lam0 - win_half) & (wave_obs <= lam0 + win_half)
    m_side = m_win & ((wave_obs < lam0 - 12.0) | (wave_obs > lam0 + 12.0))

    if int(np.sum(m_win)) < 5:
        return np.full(arr.shape[0], np.nan, dtype=np.float64)

    w_line = wave_obs[m_win]
    f_line = arr[:, m_win]
    if int(np.sum(m_side)) >= 5:
        cont = np.nanmedian(arr[:, m_side], axis=1)
    else:
        cont = np.nanmedian(f_line, axis=1)

    line = np.clip(f_line - cont[:, None], 0.0, None)
    flux = np.trapz(line, w_line, axis=1)
    return flux.astype(np.float64)


def load_target_context(extractor: PairExtractor, matches_csv: str, row_idx: int):
    desi_tab = extractor.desi_table()
    matches_df = pd.read_csv(matches_csv)
    matches_df = matches_df[matches_df["TARGETID"].isin(set(desi_tab["TARGETID"].values))].reset_index(drop=True)

    if row_idx < 0 or row_idx >= len(matches_df):
        raise IndexError(f"row_idx {row_idx} out of range for matches_df size {len(matches_df)}")

    row = matches_df.iloc[row_idx]
    targetid = int(row["TARGETID"])
    hit = desi_tab.loc[desi_tab["TARGETID"] == targetid]
    if len(hit) == 0:
        raise RuntimeError(f"TARGETID {targetid} missing in DESI table")

    desi_index = int(hit.index[0])
    desi_row = desi_tab.iloc[desi_index]
    target_ra = float(desi_row["TARGET_RA"])
    target_dec = float(desi_row["TARGET_DEC"])
    z_obj = float(desi_row["Z"])

    plateifu, sep = extractor.match_manga(target_ra, target_dec)

    loader = extractor.get_loader(desi_index)
    batch = next(iter(loader))
    maps = extractor.get_maps(plateifu)
    truth_ha = np.asarray(maps["ha_flux"], dtype=np.float64)

    maps_path = extractor._maps_path(plateifu)
    with fits.open(maps_path) as h:
        wcs_map = WCS(h["SPX_MFLUX"].header).celestial

    ny, nx = truth_ha.shape
    iy, ix = np.indices((ny, nx))
    ra_grid, dec_grid = wcs_map.pixel_to_world_values(ix.ravel(), iy.ravel())

    x_off = (
        -1.0 * (ra_grid - target_ra) * np.cos(np.deg2rad(target_dec)) * 3600.0 / PIX_SCALE
    ).reshape(ny, nx)
    y_off = (
        -1.0 * (dec_grid - target_dec) * 3600.0 / PIX_SCALE
    ).reshape(ny, nx)

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
        "truth": truth_ha,
        "coords": coords,
        "yx_pairs": yx_pairs,
    }


def get_rank_info():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    return rank, world, local_rank


def merge_outputs(output_dir: Path):
    parts = sorted(output_dir.glob("checkpoint_metrics_rank*.csv"))
    if not parts:
        raise RuntimeError(f"No rank metric files found in {output_dir}")

    dfs = [pd.read_csv(p) for p in parts]
    full = pd.concat(dfs, axis=0, ignore_index=True)
    full = full.sort_values(["epoch", "checkpoint"]).reset_index(drop=True)

    merged_csv = output_dir / "checkpoint_metrics_all.csv"
    full.to_csv(merged_csv, index=False)

    best = full.sort_values("cosine_norm", ascending=False).head(20)
    best_csv = output_dir / "checkpoint_metrics_top20.csv"
    best.to_csv(best_csv, index=False)

    summary = {
        "n_checkpoints": int(len(full)),
        "n_beats_linear": int(np.sum(full["beats_linear_baseline"].astype(bool))),
        "best_checkpoint": str(best.iloc[0]["checkpoint"]) if len(best) > 0 else None,
        "best_cosine": float(best.iloc[0]["cosine_norm"]) if len(best) > 0 else None,
        "merged_csv": str(merged_csv),
        "top20_csv": str(best_csv),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Parallel checkpoint sweep for Halpha maps on one target object")
    ap.add_argument("--checkpoint-dir", type=str, default="/pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsSimple")
    ap.add_argument("--min-epoch", type=int, default=70)
    ap.add_argument("--matches-csv", type=str, default="/pscratch/sd/p/pzehao/desi_manga_matches.csv")
    ap.add_argument("--target-row-idx", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--output-dir", type=str, default="/pscratch/sd/p/pzehao/MultiModal/notebooks/ckpt_halpha_sweep")
    ap.add_argument("--baseline-metrics-csv", type=str, default="")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--max-checkpoints", type=int, default=0)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_outputs(out_dir)
        return

    rank, world, local_rank = get_rank_info()

    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "epoch=*.ckpt")))
    ckpts = [p for p in ckpts if parse_epoch(p) > args.min_epoch]
    if args.max_checkpoints > 0:
        ckpts = ckpts[: args.max_checkpoints]
    if len(ckpts) == 0:
        raise RuntimeError("No checkpoints matched filter")

    assigned = ckpts[rank::world]

    if args.force_cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        ndev = torch.cuda.device_count()
        if ndev < 1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{local_rank % ndev}")
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True

    extractor = PairExtractor()
    ctx = load_target_context(extractor, args.matches_csv, args.target_row_idx)
    truth = ctx["truth"]
    coords = ctx["coords"]
    yx_pairs = ctx["yx_pairs"]

    x, spec, weig, error, img, img_w, img_e, z, xy_pix = ctx["batch"]

    spec = spec.to(device)
    weig = weig.to(device)
    error = error.to(device)
    img = img.to(device)
    img_w = img_w.to(device)
    img_e = img_e.to(device)
    z = z.to(device)

    wave_obs = np.asarray(extractor.wave, dtype=np.float64)
    z_obj = float(ctx["z_obj"])

    baseline_cos = np.nan
    baseline_csv = args.baseline_metrics_csv.strip()
    if baseline_csv and os.path.exists(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        if "map" in bdf.columns and "cosine_norm" in bdf.columns:
            row = bdf[bdf["map"] == "linear_baseline"]
            if len(row) > 0:
                baseline_cos = float(row.iloc[0]["cosine_norm"])

    model = build_model().to(device)
    torch.set_grad_enabled(False)

    rows = []
    t0 = time.time()

    print(
        f"RANK {rank}/{world} device={device} targetid={ctx['targetid']} "
        f"n_total_ckpt={len(ckpts)} n_assigned={len(assigned)} n_coords={len(coords)}"
    )

    for i_ckpt, ckpt in enumerate(assigned, start=1):
        ep = parse_epoch(ckpt)
        ckpt_name = Path(ckpt).name
        ck_t0 = time.time()

        chk = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(chk["state_dict"], strict=False)
        model.eval()

        out_map = np.full_like(truth, np.nan, dtype=np.float64)

        with torch.inference_mode():
            n = coords.shape[0]
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
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
                out_map[loc[:, 0], loc[:, 1]] = hvals

        cos = cosine_norm(out_map, truth)
        beats = bool(np.isfinite(baseline_cos) and np.isfinite(cos) and (cos > baseline_cos))

        map_path = ""
        if args.save_maps:
            map_path = str(out_dir / f"halpha_map_{ckpt_name.replace('.ckpt', '.npy')}")
            np.save(map_path, out_map)

        plot_path = ""
        if args.save_plots:
            plot_path = str(out_dir / f"halpha_map_{ckpt_name.replace('.ckpt', '.png')}")
            title = (
                f"targetid={ctx['targetid']} plateifu={ctx['plateifu']} "
                f"epoch={ep} cos={cos:.4f}"
            )
            plot_truth_pred_diff(truth, out_map, Path(plot_path), title)

        rows.append(
            {
                "rank": rank,
                "checkpoint": ckpt,
                "checkpoint_name": ckpt_name,
                "epoch": ep,
                "targetid": ctx["targetid"],
                "plateifu": ctx["plateifu"],
                "n_coords": int(len(coords)),
                "cosine_norm": cos,
                "baseline_cosine_norm": baseline_cos,
                "beats_linear_baseline": beats,
                "seconds": float(time.time() - ck_t0),
                "map_path": map_path,
            }
        )

        print(
            f"RANK {rank}: [{i_ckpt}/{len(assigned)}] {ckpt_name} epoch={ep} "
            f"cos={cos:.6f} beats_linear={beats} sec={time.time()-ck_t0:.2f}"
        )

    part_csv = out_dir / f"checkpoint_metrics_rank{rank:03d}.csv"
    pd.DataFrame(rows).to_csv(part_csv, index=False)

    meta = {
        "rank": rank,
        "world": world,
        "device": str(device),
        "targetid": int(ctx["targetid"]),
        "plateifu": str(ctx["plateifu"]),
        "n_assigned": len(assigned),
        "n_total": len(ckpts),
        "elapsed_sec": float(time.time() - t0),
        "part_csv": str(part_csv),
    }
    with open(out_dir / f"rank{rank:03d}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
