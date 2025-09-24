import whisper      # Модель OpenAI Whisper для транскрибации и перевода
import argparse     # Для парсинга аргументов командной строки
import os           # Для работы с файлами
from pathlib import Path  # Для удобной работы с путями
import math         # Для расчета количества чанков
import ffmpeg       # Для извлечения аудио и разрезания видео

def transcribe_and_translate(
    input_file: str,
    model_size: str = "small",
    task: str = "translate",      # "transcribe" или "translate"
    target_language: str = "ru",  # Язык перевода по умолчанию (ISO 639-1)
    output_formats: list = ["srt","txt"], # Список форматов вывода
    chunk_duration: float = 600.0 # Длительность чанка в секундах (10 минут)
):
    """
    Основная функция транскрибации и перевода файла.
    
    Аргументы:
    - input_file: путь к аудио/видео файлу, например "video.mp4" или "audio.mp3"
    - model_size: размер модели Whisper, возможные варианты: "tiny","base","small","medium","large"
    - task: "transcribe" для простой транскрибации, "translate" для перевода на английский или указанный язык
    - target_language: язык перевода по ISO 639-1, например "ru" для русского, "en" для английского
    - output_formats: список форматов вывода, возможные значения ["srt", "txt"]
    - chunk_duration: длительность одного временного чанка в секундах, используется для больших файлов
    """
    
    # Загружаем выбранную модель Whisper
    print(f"[INFO] Загружаем модель Whisper: {model_size}")
    model = whisper.load_model(model_size)

    # Получаем имя файла без расширения для формирования выходных файлов
    base_name = Path(input_file).stem

    # Получаем длительность видео через ffmpeg
    try:
        # ffmpeg.probe возвращает метаданные файла
        probe = ffmpeg.probe(input_file)
        duration = float(probe['format']['duration'])  # Длительность в секундах
        print(f"[INFO] Длительность файла: {duration:.2f} секунд")
    except Exception as e:
        print(f"[WARN] Не удалось определить длительность файла: {e}")
        duration = None  # если не удалось определить, работаем с одним чанком

    # Разбиваем на чанки
    if duration:
        num_chunks = math.ceil(duration / chunk_duration)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_duration
            end = min((i+1) * chunk_duration, duration)
            chunks.append((start, end))
    else:
        # Если длительность неизвестна, обрабатываем весь файл как один чанк
        chunks = [(0, None)]

    all_segments = []       # Список всех сегментов с таймкодами и текстом
    translated_text = ""    # Общий текст, для TXT файла

    # Обрабатываем каждый чанк по отдельности
    for idx, (start, end) in enumerate(chunks, start=1):
        print(f"[INFO] Обрабатываем чанк {idx}/{len(chunks)}: {start} - {end} секунд")

        # Создаём временный WAV-файл с текущим чанком
        tmp_chunk_file = f"tmp_chunk_{idx}.wav"

        # ffmpeg.input(...).output(...).run() выполняет конвертацию и извлечение аудио
        # ac=1 — моно (Whisper лучше с одним каналом), ar=16000 — частота дискретизации
        ffmpeg.input(input_file, ss=start, t=(None if end is None else end-start)) \
            .output(tmp_chunk_file, ac=1, ar=16000) \
            .overwrite_output() \
            .run(quiet=True)

        # Транскрибация или перевод
        # task="transcribe" — текст без перевода, "translate" — перевод на английский или target_language
        result = model.transcribe(tmp_chunk_file, task=task, language=None)  # язык источника автоопределяется

        for seg in result["segments"]:
            text = seg['text'].strip()

            # Если язык перевода не английский, выполняем перевод через модель
            if target_language != "en":
                # Используем гипотетическую функцию model.transcribe_text для перевода сегмента
                # Возвращает словарь {"text": "перевод"}
                translation = model.transcribe_text(text, task="translate", language=target_language)
                seg_text = translation["text"].strip()
            else:
                seg_text = text

            # Сохраняем текст в сегмент, корректируем тайминги относительно общего файла
            seg["text"] = seg_text
            if seg.get("start") is not None:
                seg["start"] += start
            if seg.get("end") is not None:
                seg["end"] += start

            translated_text += seg_text + " "
            all_segments.append(seg)

        # Удаляем временный файл чанка
        os.remove(tmp_chunk_file)

    # Выводим результаты в указанные форматы
    if "srt" in output_formats:
        srt_file = f"{base_name}.{target_language}.srt"
        write_srt(all_segments, srt_file)
        print(f"[OK] Субтитры сохранены: {srt_file}")

    if "txt" in output_formats:
        txt_file = f"{base_name}.{target_language}.txt"
        write_txt(translated_text, txt_file)
        print(f"[OK] Текст сохранён: {txt_file}")

    print("[INFO] Обработка завершена.")
    return output_formats


def write_srt(segments, output_file):
    """
    Создаёт SRT файл из сегментов.
    Аргументы:
    - segments: список словарей с ключами "start", "end", "text"
    - output_file: путь к файлу .srt
    """
    def fmt_time(t):
        # Преобразуем секунды в формат HH:MM:SS,mmm
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        milliseconds = int((t - int(t)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n")
            f.write(seg['text'] + "\n\n")


def write_txt(text, output_file):
    """
    Сохраняет весь текст в один TXT файл
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный скрипт транскрибации и перевода Whisper")

    # Основной входной файл (обязательный)
    parser.add_argument("input_file", help="Входной аудио/видео файл")

    # Параметры модели
    parser.add_argument("-m", "--model", default="small",
                        help="Размер модели Whisper: tiny, base, small, medium, large")

    # Задача: транскрибация или перевод
    parser.add_argument("-t", "--task", default="translate",
                        help="Задача: transcribe (только текст) или translate (перевод)")

    # Язык перевода
    parser.add_argument("-l", "--language", default="ru",
                        help="Язык перевода (ISO 639-1), по умолчанию русский")

    # Форматы вывода
    parser.add_argument("-f", "--formats", default="srt,txt",
                        help="Форматы вывода через запятую: srt,txt")

    # Длительность чанка для больших файлов
    parser.add_argument("-c", "--chunk", type=float, default=600.0,
                        help="Длительность чанка для больших файлов (сек)")

    args = parser.parse_args()

    output_formats = [fmt.strip() for fmt in args.formats.split(",")]

    transcribe_and_translate(
        input_file=args.input_file,
        model_size=args.model,
        task=args.task,
        target_language=args.language,
        output_formats=output_formats,
        chunk_duration=args.chunk
    )
