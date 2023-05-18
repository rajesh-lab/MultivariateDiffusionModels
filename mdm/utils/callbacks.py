import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


def get_checkpoint_callbacks(config):
    filename = "best_{epoch}_{step}_{val_weighted_dsm_bpd_no_offset:.2f}"
    metric = "val_weighted_dsm_bpd_no_offset"
    best_checkpoint_callback = ModelCheckpoint(
        monitor=metric, mode="min", save_top_k=2, filename=filename
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        every_n_epochs=config.eval_every_n_epoch,
        save_top_k=config.n_epochs // config.eval_every_n_epoch,
        filename="period_{epoch}",
        monitor="val_ism_bpd",
    )

    return checkpoint_callback, best_checkpoint_callback


def get_callbacks(config):
    checkpoint_callback, best_checkpoint_callback = get_checkpoint_callbacks(config)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        LogGenerationCallback(),
        checkpoint_callback,
        lr_monitor,
        ParamMonitorCallback(),
        best_checkpoint_callback,
    ]

    return callbacks


class ParamMonitorCallback(Callback):
    def on_after_backward(self, trainer, model):
        model.monitor_params_and_grads()


class LogGenerationCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0 and trainer.is_global_zero:
            gen_samples = pl_module.generate_samples(
                n_samples=pl_module.config.log_image_size
            )

            trainer.logger.experiment.log(
                {
                    "examples": [wandb.Image(x) for x in gen_samples],
                    "global_step": trainer.global_step,
                }
            )
