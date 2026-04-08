import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/pscratch/sd/p/pzehao/MultiModal")

import torch
import pytorch_lightning as pl
from lightning.pytorch import seed_everything

from models.MAE import MaskedAutoencoderViT
from utils.DataProcessing import CreateMultimodalDataLoadersIter


class ThroughputCallback(pl.Callback):
    def __init__(self, warmup_steps: int, measure_steps: int, per_device_batch_size: int):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.per_device_batch_size = per_device_batch_size
        self._base_step = 0
        self._start_t = None
        self._end_t = None
        self._reported = False

    def _report(self, trainer, measured_steps: int, rel_global_step: int):
        if not trainer.is_global_zero or self._reported:
            return

        if self._start_t is None:
            print("THROUGHPUT_RESULT_START_MISSING", json.dumps({"rel_global_step": rel_global_step}))
            self._reported = True
            return

        end_t = self._end_t if self._end_t is not None else time.perf_counter()
        dt = max(1e-12, end_t - self._start_t)

        world_size = int(getattr(trainer, "world_size", 1) or 1)
        global_batch = self.per_device_batch_size * world_size

        result = {
            "base_step": self._base_step,
            "final_step": int(trainer.global_step),
            "rel_global_step": rel_global_step,
            "world_size": world_size,
            "per_device_batch_size": self.per_device_batch_size,
            "global_batch_size": global_batch,
            "warmup_steps": self.warmup_steps,
            "measured_steps": int(measured_steps),
            "seconds": dt,
            "steps_per_s": measured_steps / dt,
            "samples_per_s_global": (measured_steps * global_batch) / dt,
            "samples_per_s_per_device": (measured_steps * self.per_device_batch_size) / dt,
        }
        print("THROUGHPUT_RESULT", json.dumps(result, sort_keys=True))
        self._reported = True

    def on_train_start(self, trainer, pl_module):
        self._base_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        rel_step = int(trainer.global_step) - self._base_step

        if self._start_t is None and rel_step >= self.warmup_steps:
            self._start_t = time.perf_counter()
            return

        if self._start_t is not None and self._end_t is None:
            measured = rel_step - self.warmup_steps
            if measured >= self.measure_steps:
                self._end_t = time.perf_counter()
                self._report(
                    trainer,
                    measured_steps=self.measure_steps,
                    rel_global_step=rel_step,
                )

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        rel_global_step = int(trainer.global_step) - self._base_step

        measured_steps = self.measure_steps
        if self._end_t is None:
            measured_steps = max(1, rel_global_step - self.warmup_steps)

        self._report(
            trainer,
            measured_steps=measured_steps,
            rel_global_step=rel_global_step,
        )


def build_model(max_epochs: int = 200):
    prob = 0.7 / 14
    patch_scheme = {
        "patch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1],
        "mask_ratios": [1.0, 13 / 14, 12 / 14, 11 / 14, 10 / 14, 9 / 14, 8 / 14, 7 / 14, 6 / 14, 5 / 14, 4 / 14, 3 / 14, 2 / 14, 1 / 14, 0.0],
        "probs": [0.3, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob],
    }

    model = MaskedAutoencoderViT(
        spec_dim=7781,
        max_epochs=max_epochs,
        warmup_epoch=1,
        mask_ratio=0.75,
        lam_img_sigma_masked=0.1,
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
        patch_scheme=patch_scheme,
    )

    # Disable expensive wandb visualizations in benchmark mode.
    model.train_vis_interval_sec = 10**12
    model._last_train_vis_time = time.time()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "block_shuffle", "interleave"], required=True)
    parser.add_argument("--end", type=int, default=4737442)
    parser.add_argument("--train-size", type=int, default=4642694)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--distributed-shard-mode",
        choices=["lightning", "contiguous"],
        default="lightning",
    )
    parser.add_argument("--train-subset-size", type=int, default=0)
    parser.add_argument("--train-block-size", type=int, default=100)
    parser.add_argument("--train-cycle-epoch", type=int, default=0)
    parser.add_argument("--train-cycle-drop-last", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=6)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsFinal/epoch=015-val_loss=-1.2884.ckpt",
    )
    args = parser.parse_args()

    seed_everything(130, workers=True)

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    loader_kwargs = dict(
        end=args.end,
        train_size=args.train_size,
        batch_size=args.batch_size,
        train_index_mode=args.mode,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        distributed_shard_mode=args.distributed_shard_mode,
    )
    if args.train_subset_size > 0:
        loader_kwargs.update(
            train_subset_size=args.train_subset_size,
            train_cycle_epoch=args.train_cycle_epoch,
            train_cycle_drop_last=args.train_cycle_drop_last,
        )
    if args.mode in ("block_shuffle", "interleave"):
        loader_kwargs.update(
            train_block_size=args.train_block_size,
        )
    if args.mode == "interleave":
        loader_kwargs.update(
            train_interleave_groups=4,
            train_interleave_span=1,
        )

    train_loader, val_loader, _ = CreateMultimodalDataLoadersIter(**loader_kwargs)
    model = build_model(max_epochs=200)

    ckpt_path = args.ckpt_path if args.ckpt_path and os.path.exists(args.ckpt_path) else None
    ckpt_global_step = 0
    if ckpt_path is not None:
        try:
            meta = torch.load(ckpt_path, map_location="cpu")
            ckpt_global_step = int(meta.get("global_step", 0))
        except Exception:
            ckpt_global_step = 0

    max_steps = ckpt_global_step + args.warmup_steps + args.measure_steps + 2

    tput_cb = ThroughputCallback(
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        per_device_batch_size=args.batch_size,
    )

    trainer = pl.Trainer(
        callbacks=[tput_cb],
        max_epochs=200,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        use_distributed_sampler=(args.distributed_shard_mode == "lightning"),
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true",
        num_nodes=args.num_nodes,
        precision="32",
        gradient_clip_val=100.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    if trainer.is_global_zero:
        print(
            f"DDP_BENCH_START mode={args.mode} end={args.end} train_size={args.train_size} "
            f"train_subset_size={args.train_subset_size} train_block_size={args.train_block_size} "
            f"train_cycle_epoch={args.train_cycle_epoch} "
            f"train_cycle_drop_last={args.train_cycle_drop_last} "
            f"batch_size={args.batch_size} warmup_steps={args.warmup_steps} measure_steps={args.measure_steps} "
            f"num_workers={args.num_workers} prefetch_factor={args.prefetch_factor} "
            f"distributed_shard_mode={args.distributed_shard_mode} "
            f"ckpt_global_step={ckpt_global_step} max_steps={max_steps}"
        )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
