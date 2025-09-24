import os
import sys
import itertools
import logging
import argparse
import traceback
from typing import List, Optional
from pathlib import Path
import subprocess

# =========================
# Настройки логирования
# =========================
ENABLE_LOG = True
LOG_FILE = "debug.log"
LOG_MODE_APPEND = True

# =========================
# Настройки по умолчанию
# =========================
KEYS_FILE = "keys.txt"
DEFAULT_CHUNK_DURATION = 600.0  # секунд
DEFAULT_LANGUAGE = "ru"
DEFAULT_ENGINE = "whisper"
DEFAULT_TRANSLATOR = "whisper"
DEFAULT_OUTPUT_FORMATS = ["txt", "srt"]

# =========================
# Инициализация логирования
# =========================
if ENABLE_LOG:
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="a" if LOG_MODE_APPEND else "w",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
    logging.info("=== Новый запуск скрипта ===")

# =========================
# Работа с API ключами
# =========================
def load_api_keys(file_path: str) -> List[str]:
    keys = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.append(line)
        logging.info(f"Загружено {len(keys)} ключей из {file_path}")
    except FileNotFoundError:
        logging.warning(f"Файл ключей {file_path} не найден. Используются переменные окружения.")
        print(f"[WARNING] Файл ключей {file_path} не найден. Используются переменные окружения.")
    return keys

api_keys = load_api_keys(KEYS_FILE)
key_cycle = itertools.cycle(api_keys)

def get_next_api_key() -> Optional[str]:
    if api_keys:
        return next(key_cycle)
    else:
        return os.getenv("OPENAI_API_KEY")

# =========================
# Импорт локальных моделей и OpenAI
# =========================
try:
    import whisper
except ImportError:
    logging.error("Библиотека openai-whisper не найдена.")
    raise

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except ImportError:
    logging.warning("Библиотека vosk не найдена.")

try:
    import openai
except ImportError:
    logging.warning("Библиотека openai не найдена.")

# =========================
# Вспомогательные функции
# =========================
def safe_openai_request(func, *args, **kwargs):
    max_attempts = len(api_keys) if api_keys else 1
    attempt = 0
    while attempt < max_attempts:
        try:
            if 'api_key' not in kwargs:
                kwargs['api_key'] = get_next_api_key()
            return func(*args, **kwargs)
        except openai.error.RateLimitError:
            logging.warning("Rate limit exceeded, переключение ключа...")
            attempt += 1
            if attempt < max_attempts:
                print(f"[INFO] Переключение на следующий API ключ (попытка {attempt+1})")
            else:
                logging.error("Все ключи исчерпаны.")
                raise
        except Exception as e:
            logging.error(f"Ошибка при запросе OpenAI: {e}")
            traceback.print_exc()
            raise

def transcribe_whisper(file_path: str, model_name: str = "small"):
    logging.info(f"Whisper транскрибация файла {file_path}, модель {model_name}")
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, fp16=False)
    return result

def transcribe_vosk(file_path: str, model_path: str):
    logging.info(f"Vosk транскрибация файла {file_path}, модель {model_path}")
    model = VoskModel(model_path)
    # Заглушка для примера
    return "Транскрибация Vosk (заглушка)"

def translate_text(text: str, target_lang: str = DEFAULT_LANGUAGE, translator: str = DEFAULT_TRANSLATOR):
    logging.info(f"Перевод текста на {target_lang} через {translator}")
    if translator == "whisper":
        return text
    elif translator in ["deepl", "google"]:
        return text
    else:
        logging.warning(f"Неизвестный переводчик {translator}")
        return text

def save_srt(text: str, srt_file: str, segments: Optional[List[dict]] = None, chunk_offset: float = 0.0):
    with open(srt_file, "a", encoding="utf-8") as f:
        if segments:
            for i, seg in enumerate(segments, 1):
                start = seg["start"] + chunk_offset
                end = seg["end"] + chunk_offset
                content = seg["text"].strip()
                start_ts = f"{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02},{int((start%1)*1000):03}"
                end_ts = f"{int(end//3600):02}:{int((end%3600)//60):02}:{int(end%60):02},{int((end%1)*1000):03}"
                f.write(f"{i}\n{start_ts} --> {end_ts}\n{content}\n\n")
        else:
            f.write(text + "\n")

def split_audio_chunks(file_path: str, chunk_duration: float) -> List[str]:
    output_files = []
    file_dir = Path(file_path).parent
    base_name = Path(file_path).stem
    output_template = file_dir / f"{base_name}_chunk%03d.mp4"
    cmd = [
        "ffmpeg", "-i", str(file_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-c", "copy",
        str(output_template)
    ]
    logging.info(f"Разбиваем файл на чанки: {cmd}")
    subprocess.run(cmd, check=True)
    i = 0
    while True:
        chunk_file = file_dir / f"{base_name}_chunk{i:03d}.mp4"
        if not chunk_file.exists():
            break
        output_files.append(str(chunk_file))
        i += 1
    return output_files

# =========================
# Главная функция
# =========================
def main():
    parser = argparse.ArgumentParser(description="Транскрибация и перевод аудио/видео")
    parser.add_argument("-i", "--input", required=True, help="Путь к файлу аудио/видео")
    parser.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Язык перевода")
    parser.add_argument("-e", "--engine", default=DEFAULT_ENGINE, choices=["whisper","vosk","gpt"], help="Движок транскрибации")
    parser.add_argument("-t", "--translator", default=DEFAULT_TRANSLATOR, help="Движок перевода")
    parser.add_argument("-f", "--formats", nargs="+", default=DEFAULT_OUTPUT_FORMATS, help="Форматы вывода")
    parser.add_argument("-m", "--model", default="small", help="Модель Whisper")
    parser.add_argument("-c", "--chunk", type=float, default=DEFAULT_CHUNK_DURATION, help="Длительность чанка в секундах")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logging.error(f"Файл {args.input} не найден")
        print(f"[ERROR] Файл {args.input} не найден")
        return

    chunk_files = [args.input]
    if args.chunk > 0:
        try:
            chunk_files = split_audio_chunks(args.input, args.chunk)
            logging.info(f"Создано {len(chunk_files)} чанков")
        except Exception as e:
            logging.warning(f"Не удалось разбить на чанки: {e}")
            chunk_files = [args.input]

    base_name = Path(args.input).stem
    all_text = ""
    if "srt" in args.formats:
        open(f"{base_name}.srt", "w").close()  # очищаем файл перед записью

    chunk_offset = 0.0
    for chunk_file in chunk_files:
        if args.engine == "whisper":
            result = transcribe_whisper(chunk_file, model_name=args.model)
            text = result["text"]
            segments = result.get("segments", [])
        elif args.engine == "vosk":
            model_path = "vosk-model-small-ru-0.22"
            text = transcribe_vosk(chunk_file, model_path)
            segments = [{"start":0, "end=args.chunk", "text":text}]  # пример таймкодов
        else:
            response = safe_openai_request(openai.ChatCompletion.create,
                                           model="gpt-4o-mini",
                                           messages=[{"role":"user","content":f"Транскрибуй файл {chunk_file}"}])
            text = response.choices[0].message.content
            segments = [{"start":0, "end":args.chunk, "text":text}]  # пример таймкодов

        translated = translate_text(text, target_lang=args.language, translator=args.translator)
        all_text += translated + "\n"
        if "srt" in args.formats:
            save_srt(translated, f"{base_name}.srt", segments, chunk_offset)
        chunk_offset += args.chunk

    if "txt" in args.formats:
        with open(f"{base_name}.txt", "w", encoding="utf-8") as f:
            f.write(all_text)

    logging.info(f"Файлы {base_name}.txt и {base_name}.srt созданы")
    print(f"Готово. Файлы {base_name}.txt и {base_name}.srt созданы.")

# =========================
# Точка входа
# =========================
if __name__ == "__main__":
    main()
