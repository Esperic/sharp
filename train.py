import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
import shutil

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    def rank_zero_only(fn):
        return fn

logger = logging.getLogger(__name__)


def build_wandb_logger(cfg):
    wandb_mode = str(getattr(cfg, "wandb", "disabled") or "disabled").lower()
    if wandb_mode in ("disabled", "false", "none"):
        return None
    if wandb_mode not in ("online", "offline"):
        raise ValueError("wandb must be one of: disabled, online, offline")

    return WandbLogger(
        project=getattr(cfg, "wandb_project", "sharp"),
        name=getattr(cfg, "tag", None),
        save_dir=getattr(cfg, "output_dir", None),
        mode=wandb_mode,
    )


@rank_zero_only
def archive_model_source(output_dir):
    shutil.copytree("src/model", f"{output_dir}/model", dirs_exist_ok=True)


def load_pretrained_weights(model, ckpt_path):
    if not ckpt_path:
        return
    logger.info(f"Warm-starting model weights from {ckpt_path}")
    incompatible = model.load_chkpt(ckpt_path)
    if incompatible is not None:
        logger.info(f"Missing keys while loading pretrained weights: {incompatible.missing_keys}")
        logger.info(f"Unexpected keys while loading pretrained weights: {incompatible.unexpected_keys}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    output_dir = cfg.output_dir
    logger.info(f"Experiments are stored in {output_dir}")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    if getattr(cfg, "checkpoint", None) and getattr(cfg, "pretrained_checkpoint", None):
        raise ValueError("Use checkpoint for resume or pretrained_checkpoint for warm-start, not both.")

    datamodule = instantiate(cfg.datamodule.pl_module, logger=logger)

    model = instantiate(cfg.model.pl_module)
    load_pretrained_weights(model, getattr(cfg, "pretrained_checkpoint", None))
    archive_model_source(output_dir)
    logger.info(model)

    callbacks = instantiate(cfg.callbacks)
    wandb_logger = build_wandb_logger(cfg)
    trainer_kwargs = dict(cfg.trainer)
    if wandb_logger is not None:
        trainer_kwargs["logger"] = wandb_logger

    trainer = pl.Trainer(
        callbacks=callbacks,
        **trainer_kwargs
    )

    fit_succeeded = False
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
        fit_succeeded = True
    finally:
        if wandb_logger is not None and trainer.is_global_zero:
            import wandb

            wandb.finish(exit_code=0 if fit_succeeded else 1)


if __name__ == "__main__":
    main()
