"""Генератор, дискриминатор и LightningModule для DCGAN."""

import os

import torch
import torchvision
from lightning import LightningModule
from torch import nn, optim
from torchvision import transforms


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


class GAN(LightningModule):
    def __init__(
        self,
        noise_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        num_samples_to_log: int = 8,
        debug_samples_epoch: int = 1,
        clearml_logger=None,
        output_dir: str = "output",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clearml_logger"])

        self.netG = Generator(noise_dim)
        self.netD = Discriminator()
        self.criterion = nn.BCELoss()

        self.automatic_optimization = False

        self.num_samples_to_log = num_samples_to_log
        self.debug_samples_epoch = debug_samples_epoch
        self.clearml_logger = clearml_logger
        self.output_dir = output_dir

        self.register_buffer("fixed_noise", torch.randn(64, noise_dim))

    def forward(self, z):
        return self.netG(z)

    def configure_optimizers(self):
        opt_g = optim.Adam(
            self.netG.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = optim.Adam(
            self.netD.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        b_size = real_images.size(0)

        opt_g, opt_d = self.optimizers()

        real_label = 1.0
        fake_label = 0.0

        # (1) Дискриминатор
        opt_d.zero_grad()
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
        output = self.netD(real_images).view(-1)
        errD_real = self.criterion(output, label)
        self.manual_backward(errD_real)
        D_x = output.mean().item()

        noise = torch.randn(b_size, self.hparams.noise_dim, device=self.device)
        fake_images = self.netG(noise)
        label.fill_(fake_label)
        output = self.netD(fake_images.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        self.manual_backward(errD_fake)
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        opt_d.step()

        # (2) Генератор
        opt_g.zero_grad()
        label.fill_(real_label)
        output = self.netD(fake_images).view(-1)
        errG = self.criterion(output, label)
        self.manual_backward(errG)
        D_G_z2 = output.mean().item()
        opt_g.step()

        self.log("Loss_D", errD, prog_bar=True, on_step=True, on_epoch=True)
        self.log("Loss_G", errG, prog_bar=True, on_step=True, on_epoch=True)
        self.log("D(x)", D_x, prog_bar=False, on_step=True, on_epoch=True)
        self.log("D(G(z))", D_G_z2, prog_bar=False, on_step=True, on_epoch=True)

        if self.clearml_logger is not None:
            self.clearml_logger.report_scalar("Loss", "D", errD.item(), self.global_step)
            self.clearml_logger.report_scalar("Loss", "G", errG.item(), self.global_step)
            self.clearml_logger.report_scalar("D(x)", "mean", D_x, self.global_step)
            self.clearml_logger.report_scalar("D(G(z))", "mean", D_G_z2, self.global_step)

    def on_train_epoch_end(self):
        step = max(1, int(self.debug_samples_epoch))
        if self.current_epoch % step != 0:
            return

        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()

        os.makedirs(self.output_dir, exist_ok=True)
        torchvision.utils.save_image(
            fake,
            os.path.join(self.output_dir, f"fake_samples_epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=8,
        )
        self.print(f"Saved samples for epoch {self.current_epoch}")

        if self.clearml_logger is not None:
            n = min(self.num_samples_to_log, fake.size(0))
            for i in range(n):
                single = transforms.ToPILImage()(fake[i])
                self.clearml_logger.report_image(
                    title="Generated Samples",
                    series=f"sample_{i}",
                    iteration=self.current_epoch,
                    image=single,
                )