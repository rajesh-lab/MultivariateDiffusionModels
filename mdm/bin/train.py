import argparse
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import click
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from mdm.datamodules import (
    CIFAR10DataModule,
    MNISTDataModule,
    CelebA256DataModule,
    ImageNetDataModule,
)
from mdm.utils.utils import get_config, print_config_nicely
from mdm.models import (
    SDELib,
    VPSDE,
    LearnedSDE,
    CLD,
    ALDA,
    MALDA,
)
from mdm.utils.callbacks import (
    LogGenerationCallback,
    ParamMonitorCallback,
)


def get_model(config: argparse.ArgumentParser) -> SDELib:
    if config.sde_type == "vpsde":
        model = VPSDE(config=config)
    elif config.sde_type == "cld":
        model = CLD(config=config)
    elif config.sde_type == "alda":
        model = ALDA(config=config)
    elif config.sde_type == "malda":
        model = MALDA(config=config)
    elif config.sde_type in ["learned", "learned_2", "learned_3"]:
        model = LearnedSDE(config=config)
    else:
        raise NotImplementedError(f"SDE {config.sde_type} not implemented")

    if config.resume_checkpoint:
        print(f"Resuming from checkpoint {config.checkpoint_path}")
        model = model.load_from_checkpoint(config.checkpoint_path)
        resume_from_checkpoint = config.checkpoint_path
    else:
        resume_from_checkpoint = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, resume_from_checkpoint


def get_datamodule(config):
    if config.dataset == "mnist":
        return MNISTDataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
        )
    elif config.dataset == "cifar":
        return CIFAR10DataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
        )
    elif config.dataset == "celeba":
        return CelebA256DataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
        )
    elif config.dataset == "imagenet":
        return ImageNetDataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
            image_width=config.width,
        )
    else:
        raise NotImplementedError(f"Data {config.dataset} not implemented")


def train(config, offline, debug_mode):
    model, resume_from_checkpoint = get_model(config)
    datamodule = get_datamodule(config)

    wandb_logger = WandbLogger(
        project=config.wandb_project,
        save_dir=config.save_dir,
        log_model=True if not offline else False,
        entity=config.wandb_entity,
        offline=offline,
        dir=config.save_dir,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_weighted_dsm_bpd_no_offset",
        mode="min",
        save_top_k=2,
        filename="best_{epoch}_{step}_{val_weighted_dsm_bpd_no_offset:.2f}",
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        every_n_epochs=config.eval_every_n_epoch,
        save_top_k=config.n_epochs // config.eval_every_n_epoch,
        filename="period_{epoch}",
        monitor="val_weighted_dsm_bpd_no_offset",
    )
    callbacks = [
        LogGenerationCallback(),
        checkpoint_callback,
        lr_monitor,
        ParamMonitorCallback(),
        best_checkpoint_callback,
    ]

    print("Starting train")
    if debug_mode:
        print("*" * 69, "\n")
        print("DEBUG MODE")
        print("*" * 69)

        debug_dict = dict(
            num_sanity_val_steps=2,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            max_epochs=100,
            profiler="advanced",
        )
    else:
        debug_dict = {"max_epochs": config.n_epochs}

    trainer = pl.Trainer(
        logger=wandb_logger,
        devices=1,
        accelerator="auto",
        deterministic=True,
        callbacks=callbacks,
        gradient_clip_val=model.config.grad_clip_val,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=config.eval_every_n_epoch,
        accumulate_grad_batches=model.config.accumulate_grad_batches,
        **debug_dict,
    )
    print(f"Gradient clip set to {trainer.gradient_clip_val}")

    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        ckpt_path=resume_from_checkpoint,
    )


@click.command()
@click.option("--config_path", required=True, help="Config file path.")
@click.option(
    "--offline",
    type=bool,
    default=False,
    required=False,
    help="run wandb in offline mode.",
)
@click.option(
    "--debug_mode",
    type=bool,
    default=False,
    required=False,
    help="run code in debug mode.",
)
def main(config_path, offline, debug_mode):
    config = get_config(config_path=config_path)

    pl.seed_everything(config.seed)
    print_config_nicely(config)

    train(config=config, offline=offline, debug_mode=debug_mode)


if __name__ == "__main__":
    main()
