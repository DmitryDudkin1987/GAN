# DCGAN on MNIST

Учебный проект: генеративно-состязательная сеть (DCGAN) для генерации цифр MNIST.
Обучение на PyTorch Lightning, логирование экспериментов — в ClearML.

## Структура

```
.
├── config.py       # CLI, гиперпараметры, инициализация ClearML
├── data.py         # MNISTDataModule
├── models.py       # Generator, Discriminator, GAN(LightningModule)
├── train.py        # точка входа
├── requirements.txt
├── .env.example
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Настройка ClearML

Скопируйте пример окружения и заполните ключи:

```bash
cp .env.example .env
# отредактируйте .env своими CLEARML_API_ACCESS_KEY / SECRET
```

Реальный `.env` в git не коммитится (см. `.gitignore`).

## Запуск

```bash
python train.py --epoch 10 --debug_samples_epoch 1
```

Артефакты:

- `checkpoints/` — чекпоинты (`ModelCheckpoint`, топ-3 по `Loss_G`)
- `output/` — сгенерированные изображения по эпохам
- ClearML-проект `My_Homework` / таск `Unit_3` — скаляры лоссов, сэмплы, чекпоинты