# DCGAN on MNIST

Учебный проект: генеративно-состязательная сеть (DCGAN) для генерации цифр MNIST.
Обучение на PyTorch Lightning, логирование экспериментов — в ClearML.

## Структура проекта

.
├── config.py       # CLI и гиперпараметры
├── data.py         # MNISTDataModule
├── models.py       # Generator, Discriminator, GAN(LightningModule)
├── train.py        # точка входа + инициализация ClearML
├── requirements.txt
└── README.md

## Установка и запуск (Quick Start)

Инструкция подходит для Linux, macOS и Windows. Примеры команд даны для терминала Linux/macOS.

### 1. Клонирование репозитория и переход в папку

git clone https://github.com/DmitryDudkin1987/GAN.git
cd GAN

(Если у вас нет git, скачайте архив с GitHub через кнопку Code -> Download ZIP, распакуйте и перейдите в папку через cd путь/к/папке/GAN).

### 2. Создание и активация виртуального окружения

Рекомендуется использовать изолированное окружение:

python3 -m venv .venv
source .venv/bin/activate

(Для Windows: .venv\Scripts\activate)

ВАЖНО (Ubuntu/Debian): Если при создании окружения вы видите ошибку "ensurepip is not available", установите системный пакет:
sudo apt update && sudo apt install python3-venv
После установки удалите старую папку (rm -rf .venv) и повторите шаг 2.

### 3. Установка зависимостей

Обновите pip и установите необходимые библиотеки:

pip install --upgrade pip
pip install -r requirements.txt

### 4. Подготовка файлов конфигурации

В текущей версии репозитория файл конфигурации может называться config.txt. Для корректного импорта в Python его ОБЯЗАТЕЛЬНО нужно переименовать:

mv config.txt config.py

(Для Windows: ren config.txt config.py)

Если вы видите в папке уже config.py, этот шаг можно пропустить.

ПРИМЕЧАНИЕ ПО CLEARML: В train.py захардкожены ключи для учебного сервера ClearML. Если сервер недоступен, обучение может зависнуть на этапе инициализации. В таком случае временно закомментируйте блок с os.environ[...] и Task.init(...) в train.py.

### 5. Запуск обучения

Запустите скрипт обучения. По умолчанию модель будет обучаться 10 эпох:

python train.py --epoch 10

Доступные аргументы командной строки:
--epoch              : Количество эпох обучения (по умолчанию 10)
--debug_samples_epoch: Как часто сохранять сгенерированные картинки (по умолчанию 1)

Пример запуска с другими параметрами:

python train.py --epoch 25 --debug_samples_epoch 5

### 6. Результаты

После запуска в корневой папке проекта появятся:
- data/               : скачанный датасет MNIST (загружается автоматически при первом запуске);
- checkpoints/        : сохраненные веса модели (лучшие 3 чекпоинта по Loss_G);
- output/             : PNG-файлы со сгенерированными цифрами после каждой эпохи;
- lightning_logs/     : логи PyTorch Lightning.
