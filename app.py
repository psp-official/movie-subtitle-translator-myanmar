import os
import re
import json
import time
import shutil
import tempfile
import subprocess

from pathlib import Path
from functools import lru_cache

from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename


# =========================================================
# CONFIGURATION
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = int(
    os.getenv("MAX_UPLOAD_MB", "500")
) * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3"
).strip()

# OCR samples one frame every N seconds.
OCR_INTERVAL_SECONDS = float(
    os.getenv("OCR_INTERVAL_SECONDS", "0.5")
)

# Audio chunks are kept short enough to reduce upload-size issues.
AUDIO_CHUNK_SECONDS = int(
    os.getenv("AUDIO_CHUNK_SECONDS", "300")
)

TRANSLATION_BATCH_SIZE = int(
    os.getenv("TRANSLATION_BATCH_SIZE", "20")
)

MAX_RETRIES = 3

app.logger.info("Movie Subtitle AI starting...")


# =========================================================
# OPTIONAL CLIENTS
# =========================================================

_gemini_client = None
_groq_client = None
_ocr_engine = None


def get_gemini_client():
    global _gemini_client

    if _gemini_client is not None:
        return _gemini_client

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY ကို Render Environment Variables "
            "ထဲမှာ ထည့်ပေးပါ။"
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai library မတပ်ဆင်ရသေးပါ။ "
            "requirements.txt ထဲမှာ google-genai ထည့်ပေးပါ။"
        ) from exc

    _gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    return _gemini_client


def get_groq_client():
    global _groq_client

    if _groq_client is not None:
        return _groq_client

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY ကို Render Environment Variables "
            "ထဲမှာ ထည့်ပေးပါ။"
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "groq library မတပ်ဆင်ရသေးပါ။ "
            "requirements.txt ထဲမှာ groq ထည့်ပေးပါ။"
        ) from exc

    _groq_client = Groq(
        api_key=GROQ_API_KEY,
        timeout=180.0,
        max_retries=2,
    )

    return _groq_client


def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR library မတပ်ဆင်ရသေးပါ။ "
            "requirements.txt ထဲမှာ rapidocr-onnxruntime "
            "ထည့်ပေးပါ။"
        ) from exc

    _ocr_engine = RapidOCR()

    return _ocr_engine


# =========================================================
# GENERAL HELPERS
# =========================================================

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg မတွေ့ပါ။ Render Environment မှာ "
            "FFmpeg တပ်ဆင်ထားဖို့ လိုအပ်ပါတယ်။"
        )


def run_ffmpeg(args, timeout=600):
    require_ffmpeg()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *args,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )

    if result.returncode != 0:
        message = (result.stderr or "FFmpeg failed")[-1500:]
        raise RuntimeError(
            "ဗီဒီယို/အသံ ပြင်ဆင်ရာတွင် ပြဿနာဖြစ်နေပါသည်။ "
            + message
        )

    return result


def get_video_info(video_path):
    require_ffmpeg()

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    if shutil.which("ffprobe") is None:
        return None

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    if result.returncode != 0:
        return None

    try:
        duration = float(result.stdout.strip())

        if duration > 0:
            return duration

    except (TypeError, ValueError):
        pass

    return None


def make_timestamp(seconds):
    seconds = max(0.0, float(seconds or 0.0))

    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    match = re.match(
        r"^\s*(\d+):([0-5]\d):([0-5]\d)[,.](\d{1,3})\s*$",
        value,
    )

    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))

    milliseconds = int(
        match.group(4).ljust(3, "0")
    )

    return (
        hours * 3600
        + minutes * 60
        + seconds
        + milliseconds / 1000
    )


def parse_time_range(value):
    if not isinstance(value, str):
        return None, None

    parts = re.split(
        r"\s*-->\s*",
        value.strip(),
        maxsplit=1,
    )

    if len(parts) != 2:
        return None, None

    return (
        parse_timestamp(parts[0]),
        parse_timestamp(parts[1]),
    )


def clean_text(value):
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text.strip()


def make_item(start, end, text, source=""):
    start = max(0.0, float(start or 0.0))
    end = max(start + 0.15, float(end or start + 2.0))

    return {
        "index": 0,
        "start": start,
        "end": end,
        "time": (
            f"{make_timestamp(start)} --> "
            f"{make_timestamp(end)}"
        ),
        "text": clean_text(text),
        "source": clean_text(source),
    }


def normalize_item(item, index=0):
    if not isinstance(item, dict):
        return None

    text = clean_text(
        item.get("text")
        or item.get("translation")
        or item.get("subtitle")
        or ""
    )

    source = clean_text(item.get("source", ""))

    raw_start = item.get(
        "start",
        item.get("start_time", item.get("begin")),
    )

    raw_end = item.get(
        "end",
        item.get("end_time", item.get("finish")),
    )

    raw_time = item.get("time", "")

    parsed_start, parsed_end = parse_time_range(
        raw_time
    )

    try:
        start = (
            float(raw_start)
            if raw_start is not None
            else parsed_start
        )
    except (ValueError, TypeError):
        start = parsed_start

    try:
        end = (
            float(raw_end)
            if raw_end is not None
            else parsed_end
        )
    except (ValueError, TypeError):
        end = parsed_end

    if start is None or end is None:
        return None

    if start < 0 or end <= start:
        return None

    result = make_item(
        start=start,
        end=end,
        text=text,
        source=source,
    )

    result["index"] = index + 1

    return result


def renumber_items(items):
    cleaned = []

    for item in items:
        if not isinstance(item, dict):
            continue

        text = clean_text(item.get("text", ""))

        if not text:
            continue

        normalized = normalize_item(
            item,
            index=len(cleaned),
        )

        if normalized is None:
            continue

        normalized["index"] = len(cleaned) + 1

        cleaned.append(normalized)

    cleaned.sort(
        key=lambda item: item["start"]
    )

    for index, item in enumerate(cleaned, start=1):
        item["index"] = index

    return cleaned


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_audio(video_path, output_dir):
    audio_path = os.path.join(
        output_dir,
        "audio.wav",
    )

    run_ffmpeg([
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio_path,
    ])

    if not os.path.exists(audio_path):
        raise RuntimeError(
            "ဗီဒီယိုထဲက အသံကို ထုတ်ယူမရပါ။"
        )

    if os.path.getsize(audio_path) < 1000:
        raise RuntimeError(
            "အသံဖိုင် မရှိပါ သို့မဟုတ် အလွန်တိုနေပါသည်။"
        )

    return audio_path


def split_audio(audio_path, output_dir):
    """
    Split long audio into manageable WAV chunks.
    Each chunk's original start offset is preserved.
    """

    require_ffmpeg()

    duration = get_video_info(audio_path)

    if not duration:
        # WAV duration probe should normally succeed.
        # If not, send the original file as one chunk.
        return [(audio_path, 0.0)]

    chunk_seconds = max(
        60,
        AUDIO_CHUNK_SECONDS,
    )

    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    chunks = []

    offset = 0.0
    index = 0

    while offset < duration:
        output_path = os.path.join(
            output_dir,
            f"audio_chunk_{index:04d}.wav",
        )

        length = min(
            chunk_seconds,
            duration - offset,
        )

        run_ffmpeg([
            "-ss", str(offset),
            "-i", str(audio_path),
            "-t", str(length),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_path,
        ])

        if os.path.exists(output_path):
            if os.path.getsize(output_path) > 1000:
                chunks.append(
                    (output_path, offset)
                )

        offset += length
        index += 1

    if not chunks:
        raise RuntimeError(
            "အသံကို အပိုင်းပိုင်းခွဲထုတ်မရပါ။"
        )

    return chunks


# =========================================================
# GROQ WHISPER TRANSCRIPTION
# =========================================================

def get_segment_value(segment, key, default=None):
    if isinstance(segment, dict):
        return segment.get(key, default)

    return getattr(segment, key, default)


def transcribe_audio_chunk(audio_path, offset=0.0):
    client = get_groq_client()

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            with open(audio_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    file=(
                        os.path.basename(audio_path),
                        audio_file.read(),
                    ),
                    model=GROQ_STT_MODEL,
                    language="zh",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            raw_segments = get_segment_value(
                result,
                "segments",
                [],
            )

            items = []

            for segment in raw_segments or []:
                start = get_segment_value(
                    segment,
                    "start",
                    None,
                )

                end = get_segment_value(
                    segment,
                    "end",
                    None,
                )

                text = clean_text(
                    get_segment_value(
                        segment,
                        "text",
                        "",
                    )
                )

                if not text:
                    continue

                try:
                    start = float(start) + offset
                    end = float(end) + offset
                except (TypeError, ValueError):
                    continue

                if end <= start:
                    end = start + 0.5

                items.append(
                    make_item(
                        start,
                        end,
                        text,
                        source=text,
                    )
                )

            return items

        except Exception as exc:
            last_error = exc

            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        "Groq Whisper မှ အသံစာသားရယူမရပါ။ "
        f"အသေးစိတ်: {str(last_error)[:500]}"
    )


def transcribe_video_audio(video_path, output_dir):
    audio_path = extract_audio(
        video_path,
        output_dir,
    )

    chunks = split_audio(
        audio_path,
        output_dir,
    )

    items = []

    for chunk_path, offset in chunks:
        chunk_items = transcribe_audio_chunk(
            chunk_path,
            offset=offset,
        )

        items.extend(chunk_items)

    return renumber_items(items)


# =========================================================
# OCR SUBTITLE DETECTION
# =========================================================

def ocr_frame_text(frame):
    import numpy as np

    engine = get_ocr_engine()

    height, width = frame.shape[:2]

    # Chinese subtitles usually appear in the lower part
    # of a movie frame. Keep enough space for higher subtitles.
    crop_y = int(height * 0.45)

    cropped = frame[crop_y:height, 0:width]

    if cropped.size == 0:
        return ""

    try:
        result, _ = engine(cropped)
    except Exception:
        return ""

    if not result:
        return ""

    detections = []

    for detection in result:
        try:
            box = detection[0]
            text = clean_text(detection[1])
            score = float(detection[2])
        except (IndexError, TypeError, ValueError):
            continue

        if not text or score < 0.35:
            continue

        try:
            center_y = sum(
                float(point[1])
                for point in box
            ) / len(box)
        except Exception:
            center_y = 0.0

        detections.append(
            (center_y, text)
        )

    detections.sort(
        key=lambda item: item[0]
    )

    # Join text lines in reading order.
    return clean_text(
        " ".join(
            text for _, text in detections
        )
    )


def extract_ocr_subtitles(video_path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV မတပ်ဆင်ရသေးပါ။ "
            "requirements.txt ထဲမှာ opencv-python-headless "
            "ထည့်ပေးပါ။"
        ) from exc

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            "ဗီဒီယိုဖိုင်ကို ဖွင့်မရပါ။"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    frame_count = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    duration = (
        frame_count / fps
        if frame_count and frame_count > 0
        else None
    )

    interval = max(
        0.2,
        OCR_INTERVAL_SECONDS,
    )

    items = []

    current_text = ""
    current_start = 0.0
    previous_time = 0.0

    frame_number = 0

    try:
        while True:
            timestamp = frame_number / fps

            if (
                duration is not None
                and timestamp > duration
            ):
                break

            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000,
            )

            success, frame = capture.read()

            if not success:
                break

            text = ocr_frame_text(frame)

            # Ignore empty frames but allow a brief OCR gap
            # before deciding the current subtitle has ended.
            if text:
                normalized_text = re.sub(
                    r"\s+",
                    " ",
                    text,
                ).strip()

                if not current_text:
                    current_text = normalized_text
                    current_start = timestamp

                elif normalized_text == current_text:
                    pass

                else:
                    if timestamp > current_start:
                        items.append(
                            make_item(
                                current_start,
                                timestamp,
                                current_text,
                                source=current_text,
                            )
                        )

                    current_text = normalized_text
                    current_start = timestamp

                previous_time = timestamp

            else:
                # A blank OCR frame ends the subtitle at the
                # current sample time.
                if current_text:
                    if timestamp - previous_time > interval * 1.5:
                        items.append(
                            make_item(
                                current_start,
                                max(
                                    current_start + 0.15,
                                    previous_time + interval,
                                ),
                                current_text,
                                source=current_text,
                            )
                        )

                        current_text = ""

            frame_number = int(
                round((timestamp + interval) * fps)
            )

    finally:
        capture.release()

    if current_text:
        end_time = (
            duration
            if duration is not None
            else previous_time + interval
        )

        items.append(
            make_item(
                current_start,
                max(
                    current_start + 0.15,
                    end_time,
                ),
                current_text,
                source=current_text,
            )
        )

    return renumber_items(items)


# =========================================================
# GEMINI TRANSLATION
# =========================================================

def parse_json_array(response_text):
    text = clean_text(response_text)

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if isinstance(data.get("translations"), list):
                return data["translations"]

            if isinstance(data.get("items"), list):
                return data["items"]

    except json.JSONDecodeError:
        pass

    # Attempt to recover a JSON array from surrounding text.
    match = re.search(
        r"\[[\s\S]*\]",
        text,
    )

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, list):
                return data

        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Gemini မှ မှန်ကန်သော JSON ဘာသာပြန်ရလဒ် မရရှိပါ။"
    )


def translate_batch(texts):
    if not texts:
        return []

    client = get_gemini_client()

    payload = [
        {
            "id": index,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]

    prompt = """
You are a professional Chinese-to-Myanmar subtitle translator.

Translate each Chinese subtitle into natural, conversational
Myanmar Burmese suitable for movie recap and dialogue subtitles.

Rules:
1. Keep every input id exactly unchanged.
2. Return exactly one translation for every input item.
3. Do not omit any items.
4. Preserve the original meaning, emotion, names, and context.
5. Use natural spoken Burmese, not stiff word-for-word Burmese.
6. Do not add explanations, notes, or quotation marks.
7. Do not return the original Chinese as the translation.
8. Keep each translation concise enough for subtitles.
9. Return valid JSON only, using this schema:
[
  {"id": 0, "text": "မြန်မာဘာသာပြန်"},
  {"id": 1, "text": "မြန်မာဘာသာပြန်"}
]

Input:
""" + json.dumps(
        payload,
        ensure_ascii=False,
    )

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            response_text = getattr(
                response,
                "text",
                None,
            )

            if not response_text:
                raise RuntimeError(
                    "Gemini က စာသားရလဒ် မပြန်ပေးပါ။"
                )

            parsed = parse_json_array(
                response_text
            )

            translations = {
                int(item["id"]): clean_text(item["text"])
                for item in parsed
                if isinstance(item, dict)
                and "id" in item
                and "text" in item
            }

            output = []

            for index in range(len(texts)):
                translation = translations.get(
                    index,
                    "",
                )

                if not translation:
                    raise RuntimeError(
                        f"စာတန်း {index + 1} အတွက် "
                        "ဘာသာပြန်ချက် မရရှိပါ။"
                    )

                output.append(translation)

            return output

        except Exception as exc:
            last_error = exc

            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        "Gemini ဘာသာပြန်မှု မအောင်မြင်ပါ။ "
        f"အသေးစိတ်: {str(last_error)[:500]}"
    )


def translate_items(items):
    if not items:
        return []

    output = []

    # Translation cache avoids repeated requests for identical
    # source text within the current processing run.
    cache = {}

    batch_size = max(
        1,
        TRANSLATION_BATCH_SIZE,
    )

    for offset in range(0, len(items), batch_size):
        batch = items[
            offset:offset + batch_size
        ]

        texts_to_translate = []
        original_positions = []

        for position, item in enumerate(batch):
            source = clean_text(
                item.get("source")
                or item.get("text")
            )

            if not source:
                continue

            if source in cache:
                continue

            original_positions.append(
                (position, source)
            )

            texts_to_translate.append(source)

        if texts_to_translate:
            translated = translate_batch(
                texts_to_translate
            )

            for source, translation in zip(
                texts_to_translate,
                translated,
            ):
                cache[source] = translation

        for item in batch:
            source = clean_text(
                item.get("source")
                or item.get("text")
            )

            if not source:
                continue

            translation = cache.get(
                source,
                "",
            )

            if not translation:
                raise RuntimeError(
                    "ဘာသာပြန်စာသား မရရှိပါ။ "
                    "တရုတ်စာသားကို မူရင်းအတိုင်း "
                    "မြန်မာစာတန်းအဖြစ် မထည့်ထားပါ။"
                )

            output.append(
                make_item(
                    start=item["start"],
                    end=item["end"],
                    text=translation,
                    source=source,
                )
            )

    return renumber_items(output)


# =========================================================
# SRT EXPORT
# =========================================================

def build_srt(items):
    normalized = renumber_items(items)

    blocks = []

    for index, item in enumerate(
        normalized,
        start=1,
    ):
        blocks.append(
            f"{index}\n"
            f"{make_timestamp(item['start'])} --> "
            f"{make_timestamp(item['end'])}\n"
            f"{item['text']}"
        )

    return (
        "\n\n".join(blocks) + "\n"
        if blocks
        else ""
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def index():
    # Flask serves templates/index.html automatically.
    from flask import render_template

    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Movie Subtitle AI",
    })


# =========================================================
# PROCESS VIDEO API
# =========================================================

@app.route("/api/process", methods=["POST"])
def process_video():
    upload = request.files.get("video")

    if upload is None:
        return jsonify({
            "error": "ဗီဒီယိုဖိုင် မရရှိပါ။"
        }), 400

    if not upload.filename:
        return jsonify({
            "error": "ဗီဒီယိုဖိုင်ကို ရွေးပေးပါ။"
        }), 400

    if not allowed_file(upload.filename):
        return jsonify({
            "error": (
                "မထောက်ပံ့သော ဖိုင်အမျိုးအစားဖြစ်ပါသည်။ "
                "MP4, MOV, MKV, WEBM သို့မဟုတ် AVI ကို သုံးပါ။"
            )
        }), 400

    mode = clean_text(
        request.form.get("mode", "auto")
    ).lower()

    if mode not in {"auto", "audio", "ocr"}:
        mode = "auto"

    safe_name = secure_filename(
        upload.filename
    ) or "uploaded_video.mp4"

    extension = Path(safe_name).suffix.lower()

    try:
        with tempfile.TemporaryDirectory(
            prefix="movie_subtitle_"
        ) as temp_dir:

            video_path = os.path.join(
                temp_dir,
                "input" + extension,
            )

            upload.save(video_path)

            if (
                not os.path.exists(video_path)
                or os.path.getsize(video_path) == 0
            ):
                return jsonify({
                    "error": "ဗီဒီယိုဖိုင် အလွတ်ဖြစ်နေပါသည်။"
                }), 400

            if mode == "audio":
                source_items = transcribe_video_audio(
                    video_path,
                    temp_dir,
                )

            elif mode == "ocr":
                source_items = extract_ocr_subtitles(
                    video_path
                )

            else:
                # Auto: try OCR first. If no subtitle text
                # is detected, use audio transcription.
                source_items = extract_ocr_subtitles(
                    video_path
                )

                if not source_items:
                    source_items = transcribe_video_audio(
                        video_path,
                        temp_dir,
                    )

            source_items = renumber_items(
                source_items
            )

            if not source_items:
                return jsonify({
                    "error": (
                        "စာတန်းထိုး သို့မဟုတ် စကားပြောအသံ "
                        "မတွေ့ပါ။ Audio mode နဲ့ ထပ်စမ်းကြည့်ပါ။"
                    )
                }), 422

            translated_items = translate_items(
                source_items
            )

            if not translated_items:
                return jsonify({
                    "error": "ဘာသာပြန်ရလဒ် မရရှိပါ။"
                }), 422

            return jsonify({
                "success": True,
                "count": len(translated_items),
                "items": translated_items,
            })

    except subprocess.TimeoutExpired:
        app.logger.exception(
            "FFmpeg operation timed out"
        )

        return jsonify({
            "error": (
                "ဗီဒီယိုလုပ်ဆောင်ချိန် ကျော်လွန်သွားပါသည်။ "
                "ဖိုင်တိုတစ်ခုနဲ့ ထပ်စမ်းကြည့်ပါ။"
            )
        }), 504

    except Exception as exc:
        app.logger.exception(
            "Video processing failed"
        )

        return jsonify({
            "error": str(exc)[:1500]
        }), 500


# =========================================================
# EXPORT API
# =========================================================

@app.route("/api/export", methods=["POST"])
def export_srt():
    data = request.get_json(
        silent=True
    ) or {}

    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify({
            "error": "items သည် list ဖြစ်ရပါမည်။"
        }), 400

    srt_content = build_srt(items)

    if not srt_content.strip():
        return jsonify({
            "error": (
                "SRT ထုတ်ရန် မှန်ကန်သော စာတန်းနှင့် "
                "အချိန်တံဆိပ် မရှိပါ။"
            )
        }), 400

    return Response(
        "\ufeff" + srt_content,
        mimetype="application/x-subrip; charset=utf-8",
        headers={
            "Content-Disposition":
                'attachment; filename="myanmar_subtitles.srt"'
        },
    )


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": (
            "ဗီဒီယိုဖိုင် အရွယ်အစား အလွန်ကြီးနေပါသည်။ "
            f"အများဆုံး {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB "
            "အထိသာ တင်နိုင်ပါသည်။"
        )
    }), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "API endpoint မတွေ့ပါ။"
        }), 404

    return "Page not found", 404


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception(
        "Unhandled server error"
    )

    return jsonify({
        "error": (
            "Server ဘက်မှာ ပြဿနာဖြစ်နေပါသည်။ "
            "Render Logs ကို စစ်ဆေးပါ။"
        )
    }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    port = int(
        os.getenv("PORT", "5000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
