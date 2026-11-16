"""Entry point: обучение DCGAN на MNIST с логированием в ClearML."""

import os

from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from config import build_config, parse_args
from data import MNISTDataModule
from models import GAN

# ---------------------------------------------------------------------------
# ClearML: временный сервер, ключи захардкожены (учебный проект)
# ---------------------------------------------------------------------------
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

os.environ["CLEARML_WEB_HOST"] = "http://51.250.6.178:8080"
os.environ["CLEARML_API_HOST"] = "http://51.250.6.178:8008"
os.environ["CLEARML_FILES_HOST"] = "http://51.250.6.178:8081"

os.environ["CLEARML_API_ACCESS_KEY"] = "YWEBBYV6ZO3P198UJ7HRIC23Z9NBBE"
os.environ["CLEARML_API_SECRET_KEY"] = (
    "VWYmCQf3D3MfkSDFKOU1bvlU5_URaMn1iMUwXRVoxXv6MKu9KbAWrMeAx52Rb210XGo"
)


def main():
    args = parse_args()
    config = build_config(args)

    task = Task.init(project_name="My_Homework", task_name="Unit_3")
    task.add_tags(["GAN"])
    logger = task.get_logger()

    config = task.connect(config)
    print("Hyperparameters:", config)

    dm = MNISTDataModule(
        batch_size=config["batch_size"],
        num_workers=2,
        val_size=config["val_size"],
    )

    model = GAN(
        noise_dim=config["noise_dim"],
        lr=config["lr"],
        b1=config["b1"],
        b2=config["b2"],
        num_samples_to_log=config["num_samples_to_log"],
        debug_samples_epoch=config["debug_samples_epoch"],
        clearml_logger=logger,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gan-{epoch:02d}-{Loss_G:.4f}",
        monitor="Loss_G",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = Trainer(
        max_epochs=config["num_epochs"],
        accelerator="auto",
        devices=1,
        log_every_n_steps=config["log_every_n_steps"],
        callbacks=[checkpoint_callback],
    )

    try:
        trainer.fit(model, datamodule=dm)

        if os.path.isdir("checkpoints"):
            task.upload_artifact("checkpoints", artifact_object="checkpoints")
        if checkpoint_callback.best_model_path:
            task.upload_artifact(
                "best_checkpoint", artifact_object=checkpoint_callback.best_model_path
            )
            print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    finally:
        task.close()


if __name__ == "__main__":
    main()