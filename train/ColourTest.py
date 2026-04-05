import os
import sys

sys.path.append("..")

import torch
import wandb
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.MAE import MaskedAutoencoderViT
from utils.DataProcessing import CreateMultimodalDataLoadersIter

TRAIN_NUM_NODES = 4
TRAIN_DEVICES = 4


if __name__ == "__main__":
    seed_everything(130, workers=True)

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            print("Enabled flash SDPA")
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("Enabled memory-efficient SDPA")

    total_target_epochs = 200
    checkpoint_dir = os.path.join(os.environ["SCRATCH"], "DESIMAE/ProductionCheckpointsFinal")
    resume_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(resume_ckpt_path):
        ckpt_meta = torch.load(resume_ckpt_path, map_location="cpu")
        last_finished_epoch = int(ckpt_meta.get("epoch", -1))
        next_epoch = last_finished_epoch + 1
        run_max_epochs = min(total_target_epochs, last_finished_epoch + 2)
        ckpt_path = resume_ckpt_path
        print(f"Resuming from {resume_ckpt_path} (last_finished_epoch={last_finished_epoch})")
    else:
        next_epoch = 0
        run_max_epochs = 1
        ckpt_path = None
        print("No previous checkpoint found. Starting from scratch for epoch 0.")

    # Keep the highest-throughput dataloader settings validated on 4x4 GPUs.
    loader_cfg = {
        "end": 4737442,
        "train_size": 4642694,
        "batch_size": 32,
        "train_index_mode": "interleave",
        "train_block_size": 100,
        "train_interleave_groups": 4,
        "train_interleave_span": 1,
        "train_subset_size": 50000,
        "train_cycle_epoch": next_epoch,
        "train_cycle_drop_last": False,
        "num_workers": 7,
        "prefetch_factor": 4,
        "distributed_shard_mode": "contiguous",
    }

    train_loader, val_loader, test_loader = CreateMultimodalDataLoadersIter(**loader_cfg)
    print("Data config:", loader_cfg)
    # train 98%, val 1%, test 1%

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=1,
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_weights_only=False,
    )

    os.environ["WANDB_DIR"] = os.environ["SCRATCH"]
    os.environ["WANDB_CACHE_DIR"] = os.path.join(os.environ["SCRATCH"], ".cache", "wandb")

    wandb.finish()

    # run_id_path = os.path.join(checkpoint_dir, "wandb_run_id.txt")
    # if os.path.exists(run_id_path):
    #     with open(run_id_path, "r", encoding="utf-8") as f:
    #         wandb_run_id = f.read().strip()
    # else:
    #     # Atomically create a shared run-id file so all distributed ranks use one run.
    #     candidate_id = wandb.util.generate_id()
    #     try:
    #         fd = os.open(run_id_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    #         with os.fdopen(fd, "w", encoding="utf-8") as f:
    #             f.write(candidate_id + "\n")
    #         wandb_run_id = candidate_id
    #     except FileExistsError:
    #         with open(run_id_path, "r", encoding="utf-8") as f:
    #             wandb_run_id = f.read().strip()

    # if not wandb_run_id:
    #     raise RuntimeError(f"Invalid W&B run id in {run_id_path}")

    # logger = WandbLogger(
    #     project="Production",
    #     name="Final Everything",
    #     id=wandb_run_id,
    #     resume="allow",
    #     log_model=True,
    # )

    logger = WandbLogger(
        project="Production",
        name="Test",
        log_model=False,
    )

    # print(f"W&B run id: {wandb_run_id} (stored at {run_id_path})")
    print(f"W&B dashboard: {logger.experiment.url}")

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    train_shard_mode = loader_cfg["distributed_shard_mode"]

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=run_max_epochs,
        logger=logger,
        use_distributed_sampler=(train_shard_mode == "lightning"),
        accelerator="gpu",
        devices=TRAIN_DEVICES,
        strategy="ddp_find_unused_parameters_true",
        num_nodes=TRAIN_NUM_NODES,
        precision="32",
        gradient_clip_val=100.0,
        gradient_clip_algorithm="norm",
    )

    prob = 0.7 / 14
    patch_scheme={
            "patch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4, 2, 1],
            "mask_ratios": [1.0, 13/14, 12/14, 11/14, 10/14, 9/14, 8/14, 7/14, 6/14, 5/14, 4/14, 3/14, 2/14, 1/14, 0.0],
            "probs": [0.3, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob],
        }

    model = MaskedAutoencoderViT(
        spec_dim=7781,
        max_epochs=200,
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

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
