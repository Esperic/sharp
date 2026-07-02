import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
import os

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    output_dir = cfg.output_dir
    logger.info(f"Experiments are stored in {output_dir}")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    datamodule = instantiate(cfg.datamodule.pl_module, logger=logger)

    model = instantiate(cfg.model.pl_module)
    os.system('cp -a %s %s' % ('src/model', output_dir))
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

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
