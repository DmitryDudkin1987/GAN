# DCGAN on MNIST

Учебный проект: генеративно-состязательная сеть (DCGAN) для генерации цифр MNIST.
Обучение на PyTorch Lightning, логирование экспериментов — в ClearML.

## Структура
```
.
├── config.py       # CLI и гиперпараметры
├── data.py         # MNISTDataModule
├── models.py       # Generator, Discriminator, GAN(LightningModule)
├── train.py        # точка входа + инициализация ClearML
├── requirements.txt
└── README.md
```
## Установка и запуск (Quick Start)

Инструкция подходит для Linux, macOS и Windows. Примеры команд даны для терминала Linux/macOS.

1. Клонирование репозитория и переход в папку:
    ```
   git clone https://github.com/DmitryDudkin1987/GAN.git
   cd GAN
```
3. Создание и активация виртуального окружения:
```
python3 -m venv .venv
   source .venv/bin/activate      # Для Windows: .venv\Scripts\activate
```
   ВАЖНО (Ubuntu/Debian): Если при создании окружения возникает ошибка "ensurepip is not available", установите пакет:
   sudo apt update && sudo apt install python3-venv
   После установки удалите старую папку (rm -rf .venv) и повторите шаг 2.

4. Установка зависимостей:
   pip install --upgrade pip
   pip install -r requirements.txt

5. Подготовка файлов конфигурации:
   В текущей версии репозитория файл конфигурации может называться config.txt. Для корректного импорта в Python его обязательно нужно переименовать:
   mv config.txt config.py         # Для Windows: ren config.txt config.py

6. Запуск обучения:
   python train.py --epoch 10

## Аргументы командной строки

--epoch               : Количество эпох обучения (по умолчанию 10)
--debug_samples_epoch : Как часто сохранять сгенерированные картинки (по умолчанию 1)

Пример запуска с другими параметрами:
python train.py --epoch 25 --debug_samples_epoch 5

## Настройка ClearML

В train.py захардкожены ключи для учебного сервера ClearML. Если сервер недоступен и обучение зависает на этапе инициализации, временно закомментируйте блок с os.environ[...] и Task.init(...) в train.py.

## Результаты

После запуска в корневой папке проекта появятся:
- data/               : скачанный датасет MNIST (загружается автоматически при первом запуске);
- checkpoints/        : сохраненные веса модели (лучшие 3 чекпоинта по Loss_G);
- output/             : PNG-файлы со сгенерированными цифрами после каждой эпохи;
- lightning_logs/     : логи PyTorch Lightning.
- ClearML-проект My_Homework / таск Unit_3 — скаляры лоссов, сэмплы, чекпоинты (если сервер доступен).
