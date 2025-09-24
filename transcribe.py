#!/usr/bin/env python3
"""
Исправленный универсальный скрипт транскрибации + перевода.
Поддержка:
 - локальная транскрибация Whisper
 - (частичная) поддержка Vosk (требует модель)
 - транскрибация через OpenAI (gpt-4o-mini-transcribe) с ротацией ключей
 - разбиение на чанки через ffmpeg
 - генерация SRT (таймкоды) и TXT
 - логирование и чтение ключей из файла keys.txt
"""

import os
import sys
import itertools
import logging
import argparse
import traceback
import subprocess
import math
from typing import List, Optional, Tuple
from pathlib import Path

# =========================
# Настройки логирования
# =========================
ENABLE_LOG = True
LOG_FILE = "debug.log"
LOG_MODE_APPEND = True

# =========================
# Настройки по умолчанию
# =========================
KEYS_FILE = "keys.txt"                    # файл с API-ключами (один ключ в строке, # - комментарий)
DEFAULT_CHUNK_DURATION = 600.0            # сек (10 минут)
DEFAULT_LANGUAGE = "ru"
DEFAULT_ENGINE = "whisper"                # whisper | vosk | gpt
DEFAULT_TRANSLATOR = "whisper"            # whisper | google | deepl
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
# Чтение API-ключей + ротация
# =========================
def load_api_keys(file_path: str) -> List[str]:
    """Читает ключи из файла: один ключ на строку, строки, начинающиеся с '#', пропускаются."""
    keys: List[str] = []
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
key_cycle = itertools.cycle(api_keys) if api_keys else None

def get_next_api_key() -> Optional[str]:
    """Возвращает следующий ключ из файла (по кругу) либо ENV OPENAI_API_KEY."""
    if key_cycle:
        return next(key_cycle)
    return os.getenv("OPENAI_API_KEY")

# =========================
# Импорт опциональных библиотек
# =========================
# Whisper (локальная оффлайн модель)
try:
    import whisper
except Exception:
    whisper = None
    logging.warning("openai-whisper не найден. Установите 'pip install openai-whisper' если нужен Whisper.")

# Vosk (локальная оффлайн модель)
try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception:
    VoskModel = None
    KaldiRecognizer = None
    logging.warning("vosk не найден. Установите 'pip install vosk' если нужен Vosk.")

# OpenAI (для GPT транскрибации)
try:
    import openai
except Exception:
    openai = None
    logging.warning("openai SDK не найден. Установите 'pip install openai' для GPT транскрибации.")

# =========================
# Вспомогательные утилиты
# =========================
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Выполняет команду subprocess и возвращает (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def get_media_duration(file_path: str) -> Optional[float]:
    """
    Получает длительность медиафайла в секундах через ffprobe.
    Возвращает None при ошибке.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        ret, out, err = run_cmd(cmd)
        if ret == 0 and out.strip():
            return float(out.strip())
        logging.warning(f"ffprobe не вернул длительность для {file_path}: {err.strip()}")
    except Exception as e:
        logging.warning(f"Ошибка при вызове ffprobe для {file_path}: {e}")
    return None

def split_audio_chunks(file_path: str, chunk_duration: float) -> List[str]:
    """
    Разбивает входной файл на хранилищем-пригодные чанки с помощью ffmpeg (segment).
    Возвращает список путей к чанкам (mp4).
    """
    file_dir = Path(file_path).parent
    base_name = Path(file_path).stem
    output_template = file_dir / f"{base_name}_chunk%03d.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(file_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        "-c", "copy",
        str(output_template)
    ]
    logging.info(f"Запуск ffmpeg для разбиения на чанки: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg вернул ошибку при разбиении: {e}")
        raise

    # Собираем список существующих чанков
    chunks: List[str] = []
    idx = 0
    while True:
        candidate = file_dir / f"{base_name}_chunk{idx:03d}.mp4"
        if not candidate.exists():
            break
        chunks.append(str(candidate))
        idx += 1
    # Если ffmpeg не создал частей (в случае короткого файла), возвращаем оригинал
    if not chunks:
        logging.info("ffmpeg не создал чанков, возвращаем оригинальный файл как один чанк.")
        return [file_path]
    logging.info(f"Создано чанков: {len(chunks)}")
    return chunks

# =========================
# Безопасные вызовы OpenAI с ротацией ключей
# =========================
def safe_openai_request(func, *args, **kwargs):
    """
    Обёртка над вызовами OpenAI API.
    Устанавливает openai.api_key из get_next_api_key и при RateLimitError
    переключается на следующий ключ (если есть).
    """
    if openai is None:
        raise RuntimeError("openai SDK не установлен. Установите 'pip install openai' для использования GPT движка.")
    max_attempts = len(api_keys) if api_keys else 1
    attempt = 0
    last_exc = None
    while attempt < max_attempts:
        key = get_next_api_key()
        if key:
            openai.api_key = key
            logging.info("Используется OpenAI ключ: ****" + (key[-6:] if len(key) > 6 else key))
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            # Если имеется класс RateLimitError в openai, используем isinstance
            try:
                rate_limit_cls = getattr(openai, "error", None)
                is_rate = rate_limit_cls and hasattr(openai.error, "RateLimitError") and isinstance(e, openai.error.RateLimitError)
            except Exception:
                is_rate = False
            # fallback: проверяем текст ошибки
            if not is_rate and isinstance(e, Exception) and "rate limit" in str(e).lower():
                is_rate = True
            if is_rate:
                logging.warning("Rate limit exceeded для текущего ключа. Переключаем ключ.")
                attempt += 1
                continue
            # иначе пробуем завершить с ошибкой
            logging.error(f"OpenAI request failed (attempt {attempt+1}/{max_attempts}): {e}")
            raise
    # если все попытки не удались
    logging.error("Все OpenAI ключи исчерпаны или неудача выполнения запроса.")
    raise last_exc if last_exc is not None else RuntimeError("OpenAI request failed without exception info.")

# =========================
# Транскрибация движками
# =========================
def transcribe_whisper(file_path: str, model_name: str = "small") -> dict:
    """Полный результат Whisper.transcribe (dict с text и segments)"""
    if whisper is None:
        raise RuntimeError("Whisper не установлен. Установите 'pip install openai-whisper'.")
    logging.info(f"Whisper: загрузка модели {model_name}")
    model = whisper.load_model(model_name)
    logging.info(f"Whisper: транскрибируем {file_path}")
    # fp16 False — безопаснее на CPU/разных системах
    result = model.transcribe(file_path, fp16=False)
    return result

def transcribe_gpt_chunk(file_path: str, translate: bool = False, target_language: str = DEFAULT_LANGUAGE) -> List[dict]:
    """
    Транскрибуем один файл (чанк) через OpenAI Audio transcription.
    Возвращаем список сегментов (здесь — один сегмент на весь чанк).
    """
    if openai is None:
        raise RuntimeError("openai SDK не установлен. Установите 'pip install openai' для GPT транскрибации.")
    logging.info(f"GPT transcription: отправляем {file_path}")
    with open(file_path, "rb") as af:
        # Вызываем с помощью safe_openai_request, чтобы применить ротацию ключей
        resp = safe_openai_request(openai.Audio.transcriptions.create, file=af, model="gpt-4o-mini-transcribe")
    # Некоторые SDK возвращают .text, другие dict — пробуем оба варианта
    text = ""
    try:
        text = getattr(resp, "text", None) or resp.get("text", "")  # type: ignore
    except Exception:
        try:
            text = resp["text"]
        except Exception:
            text = str(resp)
    # При опции translate можно дополнительно попросить GPT перевести (через chat completion)
    if translate and target_language and target_language != "en":
        logging.info("GPT: выполняем перевод через ChatCompletion")
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        chat_resp = safe_openai_request(openai.ChatCompletion.create, model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        # извлекаем текст из chat_resp
        try:
            translated = chat_resp.choices[0].message.content
        except Exception:
            translated = getattr(chat_resp, "choices", [{}])[0].get("message", {}).get("content", "") if isinstance(chat_resp, dict) else str(chat_resp)
        text = translated or text
    # выдать сегент на весь файл; конечная привязка по offset сделается при записи SRT
    duration = get_media_duration(file_path) or 0.0
    return [{"start": 0.0, "end": float(duration), "text": text}]

def transcribe_vosk_chunk(file_path: str, model_path: str) -> List[dict]:
    """
    Транскрибация чанка через Vosk (если установлено).
    Возвращает один сегмент на весь чанк с текстом (детальное разделение - сложнее и зависит от модели).
    """
    if VoskModel is None:
        raise RuntimeError("Vosk не установлен. Установите 'pip install vosk' и скачайте модель.")
    # Для Vosk нам нужен WAV с нужной частотой — проще извлечь временный WAV и прогнать
    tmp_wav = str(Path(file_path).with_suffix(".tmp.wav"))
    try:
        # конвертация в WAV 16k mono (Vosk рекомендуется 16000)
        cmd = ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", tmp_wav]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import wave, json
        wf = wave.open(tmp_wav, "rb")
        rec = KaldiRecognizer(VoskModel(model_path), wf.getframerate())
        rec.SetWords(True)
        data = wf.readframes(wf.getnframes())
        result_text = ""
        if rec.AcceptWaveform(data):
            j = json.loads(rec.Result())
            result_text = j.get("text", "")
        else:
            j = json.loads(rec.PartialResult())
            result_text = j.get("partial", "")
        wf.close()
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
    duration = get_media_duration(file_path) or 0.0
    return [{"start": 0.0, "end": float(duration), "text": result_text}]

# =========================
# Сохранение SRT/TXT
# =========================
def save_srt_append(srt_file: str, segments: List[dict], chunk_offset: float, start_index: int) -> int:
    """
    Добавляет список сегментов в SRT файл, с учётом смещения chunk_offset.
    Возвращает следующий индекс (start_index + len(segments))
    """
    with open(srt_file, "a", encoding="utf-8") as f:
        idx = start_index
        for seg in segments:
            start = float(seg.get("start", 0.0)) + chunk_offset
            end = float(seg.get("end", 0.0)) + chunk_offset
            text = seg.get("text", "").strip()
            # формат времени HH:MM:SS,mmm
            def fmt(t: float) -> str:
                hours = int(t // 3600)
                minutes = int((t % 3600) // 60)
                seconds = int(t % 60)
                millis = int((t - int(t)) * 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            f.write(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
            idx += 1
    return start_index + len(segments)

# =========================
# Перевод (заглушки / места для интеграции)
# =========================
def translate_text(text: str, target_lang: str = DEFAULT_LANGUAGE, translator: str = DEFAULT_TRANSLATOR) -> str:
    """
    Перевод текста. Сейчас: заглушка, возвращает исходный текст.
    Здесь можно интегрировать Deepl/Google Translate/вызов GPT для перевода.
    """
    logging.info(f"translate_text: translator={translator}, target={target_lang}")
    # TODO: реализовать реальные вызовы Deepl/Google
    return text

# =========================
# CLI и основной поток
# =========================
def main():
    parser = argparse.ArgumentParser(description="Транскрибация и перевод аудио/видео (Whisper / Vosk / GPT)")
    parser.add_argument("-i", "--input", required=True, help="Путь к входному файлу (аудио/видео)")
    parser.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Язык перевода (ISO 639-1), по умолчанию 'ru'")
    parser.add_argument("-e", "--engine", default=DEFAULT_ENGINE, choices=["whisper", "vosk", "gpt"], help="Движок транскрибации")
    parser.add_argument("-t", "--translator", default=DEFAULT_TRANSLATOR, help="Переводчик (whisper/google/deepl)")
    parser.add_argument("-f", "--formats", nargs="+", default=DEFAULT_OUTPUT_FORMATS, help="Форматы вывода: txt srt")
    parser.add_argument("-m", "--model", default="small", help="Модель Whisper (tiny, base, small, medium, large)")
    parser.add_argument("-c", "--chunk", type=float, default=DEFAULT_CHUNK_DURATION, help="Длительность чанка в секундах (0 - без разбиения)")
    parser.add_argument("--vosk-model-path", default="vosk-model-small-ru-0.22", help="Путь к папке модели Vosk (если используется)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        logging.error(f"Входной файл не найден: {input_path}")
        print(f"[ERROR] Входной файл не найден: {input_path}")
        return

    # Разбиение на чанки (если указан положительный chunk)
    if args.chunk and args.chunk > 0:
        try:
            chunk_files = split_audio_chunks(input_path, args.chunk)
        except Exception as e:
            logging.warning(f"Не удалось разбить файл на чанки: {e}. Будем обрабатывать целиком.")
            chunk_files = [input_path]
    else:
        chunk_files = [input_path]

    base_name = Path(input_path).stem
    # очищаем srt файл заранее (если выбран)
    srt_file = f"{base_name}.srt"
    if "srt" in args.formats:
        open(srt_file, "w", encoding="utf-8").close()

    all_texts: List[str] = []
    srt_index = 1
    chunk_offset = 0.0  # смещение времени (в секундах) для текущего чанка

    for chunk_file in chunk_files:
        logging.info(f"Обработка чанка: {chunk_file}")
        # определяем фактическую длительность чанка
        chunk_dur = get_media_duration(chunk_file) or args.chunk or 0.0

        try:
            if args.engine == "whisper":
                # Whisper вернёт dict с 'text' и 'segments'
                whisper_result = transcribe_whisper(chunk_file, model_name=args.model)
                text = whisper_result.get("text", "")
                segments = whisper_result.get("segments", [])
            elif args.engine == "vosk":
                # Vosk: возвращаем список сегментов (у нас — один сегмент на чанк)
                segments = transcribe_vosk_chunk(chunk_file, args.vosk_model_path)
                # агрегация текста
                text = " ".join([seg.get("text", "") for seg in segments])
            else:  # gpt
                segments = transcribe_gpt_chunk(chunk_file, translate=(args.language != "en"), target_language=args.language)
                text = " ".join([seg.get("text", "") for seg in segments])
        except Exception as e:
            logging.error(f"Ошибка при транскрибации чанка {chunk_file}: {e}")
            traceback.print_exc()
            # пропускаем этот чанк, но продолжаем обработку остальных
            chunk_offset += chunk_dur
            continue

        # перевод (можно интегрировать Deepl/Google здесь)
        translated = translate_text(text, target_lang=args.language, translator=args.translator)
        all_texts.append(translated)

        # запись SRT (если требуется) — используем сегменты; если их нет, создаём одиночный сегмент
        if "srt" in args.formats:
            if not segments or len(segments) == 0:
                # создаём один сегмент на весь чанк
                segs = [{"start": 0.0, "end": float(chunk_dur), "text": translated}]
            else:
                # если сегменты есть, заменяем их текст на переведённый (сохраняем тайминги)
                segs = []
                # segments от whisper уже имеют текст — но мы заменим текст на translated chunkwise
                # более точная логика: сопоставление сегментов -> перевод каждого сегмента отдельно (можно усовершенствовать)
                for seg in segments:
                    segs.append({"start": seg.get("start", 0.0), "end": seg.get("end", 0.0), "text": seg.get("text", "").strip()})
                # NOTE: здесь мы используем оригинальные сегменты для таймингов, не переведённые тексты.
                # Если нужен перевод каждого сегмента отдельно — это дополнительный шаг (перевести seg['text'])
            # Записываем в SRT, учитывая смещение chunk_offset и текущий srt_index
            srt_index = save_srt_append(srt_file, segs, chunk_offset, srt_index)

        # сдвигаем offset на длительность чанка
        chunk_offset += chunk_dur

    # сохраняем TXT (все переведённые части)
    if "txt" in args.formats:
        txt_file = f"{base_name}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_texts))
        logging.info(f"TXT сохранён: {txt_file}")

    logging.info(f"Готово. Файлы: {base_name}.txt (если выбран), {base_name}.srt (если выбран).")
    print(f"Готово. Проверьте {base_name}.txt и {base_name}.srt (если они были выбраны).")

if __name__ == "__main__":
    main()
