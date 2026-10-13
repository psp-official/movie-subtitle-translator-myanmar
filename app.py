import os
import re
import json
import time
import shutil
import tempfile
import subprocess

from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template
from werkzeug.utils import secure_filename


# =========================================================
# CONFIGURATION
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
)

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi"
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3"
).strip()

OCR_INTERVAL_SECONDS = max(
    0.25,
    float(os.getenv("OCR_INTERVAL_SECONDS", "0.5"))
)

OCR_CONFIDENCE = max(
    0.0,
    min(1.0, float(os.getenv("OCR_CONFIDENCE", "0.45")))
)

AUDIO_CHUNK_SECONDS = max(
    60,
    int(os.getenv("AUDIO_CHUNK_SECONDS", "300"))
)

TRANSLATION_BATCH_SIZE = max(
    1,
    min(50, int(os.getenv("TRANSLATION_BATCH_SIZE", "20")))
)

MAX_RETRIES = 3

# OCR subtitle filtering
OCR_CROP_TOP_RATIO = 0.35
OCR_MIN_PERSISTENCE = 2
OCR_MAX_EMPTY_GAP = 1.0
OCR_MAX_LINES = 2

_gemini_client = None
_groq_client = None
_ocr_engine = None

app.logger.info("Movie Subtitle AI starting...")


# =========================================================
# API CLIENTS
# =========================================================

def get_gemini_client():
    global _gemini_client

    if _gemini_client is not None:
        return _gemini_client

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY ကို Render Environment Variables ထဲထည့်ပါ။"
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai library မရှိပါ။ requirements.txt ကိုစစ်ပါ။"
        ) from exc

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def get_groq_client():
    global _groq_client

    if _groq_client is not None:
        return _groq_client

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY ကို Render Environment Variables ထဲထည့်ပါ။"
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "groq library မရှိပါ။ requirements.txt ကိုစစ်ပါ။"
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
            "rapidocr-onnxruntime library မရှိပါ။"
        ) from exc

    _ocr_engine = RapidOCR()
    return _ocr_engine


# =========================================================
# GENERAL HELPERS
# =========================================================

def clean_text(value):
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text.strip()


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def require_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg မတွေ့ပါ။ Server တွင် FFmpeg တပ်ဆင်ပါ။"
        )

    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "FFprobe မတွေ့ပါ။ FFmpeg/FFprobe ကိုတပ်ဆင်ပါ။"
        )


def run_ffmpeg(args, timeout=600):
    require_ffmpeg()

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg error: "
            + (result.stderr or "Unknown error")[-1200:]
        )

    return result


def get_video_info(video_path):
    if not shutil.which("ffprobe"):
        return None

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
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
        return duration if duration > 0 else None
    except (TypeError, ValueError):
        return None


def make_timestamp(seconds):
    milliseconds = int(
        round(max(0.0, float(seconds or 0)) * 1000)
    )

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d},"
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

    hours, minutes, seconds = map(int, match.groups()[:3])
    milliseconds = int(match.group(4).ljust(3, "0"))

    return (
        hours * 3600
        + minutes * 60
        + seconds
        + milliseconds / 1000
    )


def parse_time_range(value):
    if not isinstance(value, str):
        return None, None

    parts = re.split(r"\s*-->\s*", value.strip(), maxsplit=1)

    if len(parts) != 2:
        return None, None

    return parse_timestamp(parts[0]), parse_timestamp(parts[1])


def make_item(start, end, text, source=""):
    start = max(0.0, float(start or 0.0))
    end = max(start + 0.15, float(end or start + 2.0))

    return {
        "index": 0,
        "start": start,
        "end": end,
        "time": (
            f"{make_timestamp(start)} --> {make_timestamp(end)}"
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
        "start", item.get("start_time", item.get("begin"))
    )

    raw_end = item.get(
        "end", item.get("end_time", item.get("finish"))
    )

    parsed_start, parsed_end = parse_time_range(
        item.get("time", "")
    )

    try:
        start = (
            float(raw_start)
            if raw_start is not None
            else parsed_start
        )
    except (TypeError, ValueError):
        start = parsed_start

    try:
        end = (
            float(raw_end)
            if raw_end is not None
            else parsed_end
        )
    except (TypeError, ValueError):
        end = parsed_end

    if start is None or end is None or start < 0 or end <= start:
        return None

    result = make_item(start, end, text, source)
    result["index"] = index + 1

    return result


def renumber_items(items):
    cleaned = []

    for item in items:
        if not isinstance(item, dict):
            continue

        normalized = normalize_item(item, len(cleaned))

        if normalized and normalized["text"]:
            cleaned.append(normalized)

    cleaned.sort(key=lambda item: item["start"])

    for index, item in enumerate(cleaned, start=1):
        item["index"] = index

    return cleaned


# =========================================================
# AUDIO EXTRACTION AND CHUNKING
# =========================================================

def extract_audio(video_path, output_dir):
    audio_path = os.path.join(output_dir, "audio.wav")

    run_ffmpeg([
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio_path,
    ])

    if (
        not os.path.exists(audio_path)
        or os.path.getsize(audio_path) < 1000
    ):
        raise RuntimeError("ဗီဒီယိုထဲမှ အသံကို ထုတ်ယူမရပါ။")

    return audio_path


def split_audio(audio_path, output_dir):
    duration = get_video_info(audio_path)

    if not duration:
        return [(audio_path, 0.0)]

    chunk_seconds = max(60, AUDIO_CHUNK_SECONDS)

    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    chunks = []
    offset = 0.0
    index = 0

    while offset < duration:
        output_path = os.path.join(
            output_dir,
            f"audio_chunk_{index:04d}.wav"
        )

        length = min(chunk_seconds, duration - offset)

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

        if (
            os.path.exists(output_path)
            and os.path.getsize(output_path) > 1000
        ):
            chunks.append((output_path, offset))

        offset += length
        index += 1

    if not chunks:
        raise RuntimeError("အသံဖိုင်ကို အပိုင်းပိုင်းခွဲမရပါ။")

    return chunks


# =========================================================
# STEP 1: CHINESE AUDIO -> CHINESE TEXT
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
                    prompt=(
                        "这是一段中文影视对白。请准确识别普通话语音，"
                        "特别注意人名、地名、门派名称、修炼境界和武功名称。"
                        "保留原始中文，不要翻译，不要改写，不要遗漏对白，"
                        "不要自行添加内容。"
                    ),
                    temperature=0,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            raw_segments = get_segment_value(
                result, "segments", []
            )

            items = []

            for segment in raw_segments or []:
                start = get_segment_value(segment, "start")
                end = get_segment_value(segment, "end")
                text = clean_text(
                    get_segment_value(segment, "text", "")
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
                    make_item(start, end, text, source=text)
                )

            return items

        except Exception as exc:
            last_error = exc

            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        "Groq Whisper အသံဖမ်းယူမှု မအောင်မြင်ပါ: "
        + str(last_error)[:500]
    )


def transcribe_video_audio(video_path, output_dir):
    audio_path = extract_audio(video_path, output_dir)
    chunks = split_audio(audio_path, output_dir)

    items = []

    for chunk_path, offset in chunks:
        items.extend(
            transcribe_audio_chunk(chunk_path, offset)
        )

    return renumber_items(items)


# =========================================================
# OCR: READ SUBTITLE TEXT FROM VIDEO FRAMES
# =========================================================

def _box_center_y(box):
    try:
        return sum(
            float(point[1]) for point in box
        ) / len(box)
    except Exception:
        return 0.0


def _box_center_x(box):
    try:
        return sum(
            float(point[0]) for point in box
        ) / len(box)
    except Exception:
        return 0.0


def _box_width(box):
    try:
        xs = [float(point[0]) for point in box]
        return max(xs) - min(xs)
    except Exception:
        return 0.0


def _join_detections_into_lines(detections):
    """
    Group OCR detections that appear on the same horizontal line.
    Coordinates are relative to the cropped image.
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda item: (item["center_y"], item["center_x"])
    )

    grouped = []

    for detection in detections:
        matched_group = None

        for group in grouped:
            average_y = sum(
                item["center_y"] for item in group
            ) / len(group)

            if abs(detection["center_y"] - average_y) <= 22:
                matched_group = group
                break

        if matched_group is None:
            grouped.append([detection])
        else:
            matched_group.append(detection)

    lines = []

    for group in grouped:
        group.sort(key=lambda item: item["center_x"])

        line_text = clean_text(
            " ".join(item["text"] for item in group)
        )

        if line_text:
            average_y = sum(
                item["center_y"] for item in group
            ) / len(group)

            lines.append({
                "center_y": average_y,
                "text": line_text,
                "width": sum(
                    item["width"] for item in group
                ),
            })

    return sorted(lines, key=lambda item: item["center_y"])


def ocr_frame_text(frame):
    """
    Focus on likely subtitle areas.

    Supports multiple scripts according to what the installed OCR
    model can recognize. It does not hard-code Chinese-only filtering.
    """

    engine = get_ocr_engine()

    height, width = frame.shape[:2]

    crop_top = int(height * OCR_CROP_TOP_RATIO)
    cropped = frame[crop_top:height, 0:width]

    if cropped.size == 0:
        return ""

    try:
        result, _ = engine(cropped)
    except Exception:
        app.logger.exception("OCR frame processing failed")
        return ""

    if not result:
        return ""

    detections = []

    for detection in result:
        try:
            box = detection[0]
            text = clean_text(detection[1])
            confidence = float(detection[2])
        except (IndexError, TypeError, ValueError):
            continue

        text = re.sub(r"\s+", " ", text).strip()

        if not text or confidence < OCR_CONFIDENCE:
            continue

        center_y = _box_center_y(box)
        center_x = _box_center_x(box)
        box_width = _box_width(box)

        # Ignore very small detections likely to be noise.
        if box_width < 8:
            continue

        detections.append({
            "text": text,
            "center_y": center_y,
            "center_x": center_x,
            "width": box_width,
            "confidence": confidence,
        })

    if not detections:
        return ""

    lines = _join_detections_into_lines(detections)

    if not lines:
        return ""

    # Keep only the bottom-most likely subtitle lines.
    # This reduces logos and text higher up in the frame.
    lines = lines[-OCR_MAX_LINES:]

    output = " ".join(
        line["text"] for line in lines
    )

    output = re.sub(r"\s+", " ", output).strip()

    return output


def _text_similarity(first, second):
    from difflib import SequenceMatcher

    first = re.sub(r"\s+", "", clean_text(first))
    second = re.sub(r"\s+", "", clean_text(second))

    if not first or not second:
        return 0.0

    return SequenceMatcher(None, first, second).ratio()


def extract_ocr_subtitles(video_path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV library မရှိပါ။"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError("ဗီဒီယိုဖိုင်ကို ဖွင့်မရပါ။")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    if fps <= 0:
        fps = 25.0

    frame_count_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = (
        frame_count_total / fps
        if frame_count_total > 0
        else None
    )

    interval = max(0.25, OCR_INTERVAL_SECONDS)

    items = []

    current_text = ""
    current_start = 0.0
    last_seen_time = 0.0

    pending_text = ""
    pending_count = 0
    pending_start = 0.0

    frame_number = 0

    def finish_current(end_time):
        nonlocal current_text, current_start, last_seen_time

        if not current_text:
            return

        end_time = max(
            current_start + 0.20,
            float(end_time)
        )

        items.append(
            make_item(
                current_start,
                end_time,
                current_text,
                source=current_text,
            )
        )

        current_text = ""
        current_start = 0.0
        last_seen_time = 0.0

    try:
        while True:
            timestamp = frame_number / fps

            if duration is not None and timestamp >= duration:
                break

            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000
            )

            success, frame = capture.read()

            if not success:
                break

            detected_text = ocr_frame_text(frame)
            detected_text = re.sub(
                r"\s+", " ", detected_text
            ).strip()

            if detected_text:
                if not current_text:
                    # Require repeated detections before accepting
                    # a new subtitle to reduce one-frame noise.
                    if (
                        pending_text
                        and _text_similarity(
                            pending_text, detected_text
                        ) >= 0.80
                    ):
                        pending_count += 1
                    else:
                        pending_text = detected_text
                        pending_count = 1
                        pending_start = timestamp

                    if pending_count >= OCR_MIN_PERSISTENCE:
                        current_text = detected_text
                        current_start = pending_start
                        last_seen_time = timestamp

                        pending_text = ""
                        pending_count = 0

                else:
                    similarity = _text_similarity(
                        current_text, detected_text
                    )

                    if similarity >= 0.72:
                        # Keep the current subtitle together even if
                        # OCR makes minor recognition mistakes.
                        if len(detected_text) >= len(current_text) * 0.65:
                            current_text = detected_text

                        last_seen_time = timestamp

                        pending_text = ""
                        pending_count = 0

                    else:
                        # Confirm a different subtitle over multiple
                        # samples before ending the previous subtitle.
                        if (
                            pending_text
                            and _text_similarity(
                                pending_text, detected_text
                            ) >= 0.80
                        ):
                            pending_count += 1
                        else:
                            pending_text = detected_text
                            pending_count = 1
                            pending_start = timestamp

                        if pending_count >= OCR_MIN_PERSISTENCE:
                            finish_current(pending_start)

                            current_text = detected_text
                            current_start = pending_start
                            last_seen_time = timestamp

                            pending_text = ""
                            pending_count = 0

            else:
                pending_text = ""
                pending_count = 0

                if (
                    current_text
                    and timestamp - last_seen_time > OCR_MAX_EMPTY_GAP
                ):
                    finish_current(
                        last_seen_time + interval
                    )

            frame_number = int(
                round((timestamp + interval) * fps)
            )

    finally:
        capture.release()

    if current_text:
        end_time = (
            duration
            if duration is not None
            else last_seen_time + interval
        )

        finish_current(end_time)

    # Remove duplicates and invalid subtitle segments.
    final_items = []

    for item in renumber_items(items):
        if item["end"] <= item["start"]:
            continue

        if not item["text"]:
            continue

        if final_items:
            previous = final_items[-1]

            same_text = (
                _text_similarity(
                    previous["text"], item["text"]
                ) >= 0.94
            )

            close_together = (
                item["start"] - previous["end"] <= 0.15
            )

            if same_text and close_together:
                previous["end"] = max(
                    previous["end"], item["end"]
                )
                previous["time"] = (
                    f"{make_timestamp(previous['start'])} --> "
                    f"{make_timestamp(previous['end'])}"
                )
                continue

        final_items.append(item)

    return renumber_items(final_items)


# =========================================================
# STEP 2: CHINESE TEXT -> MYANMAR TRANSLATION
# =========================================================

def parse_json_array(response_text):
    text = clean_text(response_text)

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ("translations", "items"):
                if isinstance(data.get(key), list):
                    return data[key]

    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, list):
                return data

        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Gemini မှ မှန်ကန်သော JSON ရလဒ် မရရှိပါ။"
    )


def translate_batch(texts):
    if not texts:
        return []

    client = get_gemini_client()

    payload = [
        {"id": index, "text": text}
        for index, text in enumerate(texts)
    ]

    prompt = """
You are an expert Chinese-to-Myanmar subtitle translator.

Translate Chinese dialogue into natural, conversational
Myanmar Burmese suitable for movie subtitles, especially
Chinese wuxia, xianxia, cultivation, fantasy and historical dramas.

STRICT RULES:
1. Preserve every input ID exactly.
2. Return one translation for every input item.
3. Never omit, merge, duplicate, or reorder subtitle items.
4. Understand the meaning from the Chinese source.
5. Use natural spoken Myanmar, not stiff literal translation.
6. Keep names and titles consistent.
7. Handle sect names, cultivation levels and martial arts terms carefully.
8. Do not invent dialogue or add explanations.
9. Use Myanmar script.
10. Keep each subtitle concise and easy to read.
11. Do not return the original Chinese as the translation.
12. Return valid JSON only in this format:
[
  {"id": 0, "text": "မြန်မာဘာသာပြန်"}
]

Input:
""" + json.dumps(payload, ensure_ascii=False)

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            response_text = getattr(response, "text", None)

            if not response_text:
                raise RuntimeError(
                    "Gemini က စာသားရလဒ် မပေးပါ။"
                )

            parsed = parse_json_array(response_text)
            translations = {}

            for item in parsed:
                if not isinstance(item, dict):
                    continue

                try:
                    item_id = int(item["id"])
                    translated_text = clean_text(item["text"])
                except (KeyError, TypeError, ValueError):
                    continue

                if translated_text:
                    translations[item_id] = translated_text

            output = []

            for index in range(len(texts)):
                translation = translations.get(index, "")

                if not translation:
                    raise RuntimeError(
                        f"စာတန်း {index + 1} အတွက် ဘာသာပြန်မရပါ။"
                    )

                output.append(translation)

            return output

        except Exception as exc:
            last_error = exc

            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        "Gemini ဘာသာပြန်မှု မအောင်မြင်ပါ: "
        + str(last_error)[:500]
    )


def translate_items(items):
    if not items:
        return []

    output = []
    translation_cache = {}

    batch_size = max(1, TRANSLATION_BATCH_SIZE)

    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        texts_to_translate = []

        for item in batch:
            source = clean_text(
                item.get("source") or item.get("text")
            )

            if (
                source
                and source not in translation_cache
                and source not in texts_to_translate
            ):
                texts_to_translate.append(source)

        if texts_to_translate:
            translated = translate_batch(texts_to_translate)

            for source, translation in zip(
                texts_to_translate, translated
            ):
                translation_cache[source] = translation

        for item in batch:
            source = clean_text(
                item.get("source") or item.get("text")
            )

            if not source:
                continue

            translation = translation_cache.get(source, "")

            if not translation:
                raise RuntimeError(
                    "မြန်မာဘာသာပြန်စာသား မရရှိပါ။"
                )

            output.append(
                make_item(
                    item["start"],
                    item["end"],
                    translation,
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

    for index, item in enumerate(normalized, start=1):
        blocks.append(
            f"{index}\n"
            f"{make_timestamp(item['start'])} --> "
            f"{make_timestamp(item['end'])}\n"
            f"{item['text']}"
        )

    return "\n\n".join(blocks) + ("\n" if blocks else "")


# =========================================================
# PAGES AND HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Movie Subtitle AI",
        "transcription_model": GROQ_STT_MODEL,
        "translation_model": GEMINI_MODEL,
    })


# =========================================================
# PROCESS VIDEO
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
                "MP4, MOV, MKV, WEBM သို့မဟုတ် AVI ဖိုင်ကိုသုံးပါ။"
            )
        }), 400

    mode = clean_text(
        request.form.get("mode", "auto")
    ).lower()

    if mode not in {"auto", "audio", "ocr"}:
        mode = "auto"

    safe_name = (
        secure_filename(upload.filename)
        or "uploaded_video.mp4"
    )

    extension = Path(safe_name).suffix.lower()

    try:
        with tempfile.TemporaryDirectory(
            prefix="movie_subtitle_"
        ) as temp_dir:

            video_path = os.path.join(
                temp_dir,
                "input" + extension
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
                    video_path, temp_dir
                )

            elif mode == "ocr":
                source_items = extract_ocr_subtitles(
                    video_path
                )

            else:
                # Auto mode: try video subtitles first.
                source_items = extract_ocr_subtitles(
                    video_path
                )

                # Fall back to speech recognition when OCR
                # does not find usable subtitle text.
                if not source_items:
                    source_items = transcribe_video_audio(
                        video_path, temp_dir
                    )

            source_items = renumber_items(source_items)

            if not source_items:
                return jsonify({
                    "error": (
                        "အသံစာသား သို့မဟုတ် ဗီဒီယိုစာတန်း မတွေ့ပါ။ "
                        "Audio mode နဲ့ ထပ်စမ်းပါ။"
                    )
                }), 422

            translated_items = translate_items(source_items)

            if not translated_items:
                return jsonify({
                    "error": "ဘာသာပြန်ရလဒ် မရရှိပါ။"
                }), 422

            return jsonify({
                "success": True,
                "count": len(translated_items),
                "source_language": "zh",
                "target_language": "my",
                "transcription_model": GROQ_STT_MODEL,
                "translation_model": GEMINI_MODEL,
                "source_items": source_items,
                "items": translated_items,
            })

    except subprocess.TimeoutExpired:
        app.logger.exception("FFmpeg operation timed out")

        return jsonify({
            "error": (
                "ဗီဒီယိုလုပ်ဆောင်ချိန် ကျော်လွန်သွားပါသည်။ "
                "ဖိုင်တိုတစ်ခုနဲ့ စမ်းပါ။"
            )
        }), 504

    except Exception as exc:
        app.logger.exception("Video processing failed")

        return jsonify({
            "error": str(exc)[:1500]
        }), 500


# =========================================================
# EXPORT SRT
# =========================================================

@app.route("/api/export", methods=["POST"])
def export_srt():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify({
            "error": "items သည် list ဖြစ်ရပါမည်။"
        }), 400

    srt_content = build_srt(items)

    if not srt_content.strip():
        return jsonify({
            "error": (
                "SRT ထုတ်ရန် စာတန်းနှင့် အချိန်တံဆိပ် မရှိပါ။"
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
    max_mb = (
        app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    )

    return jsonify({
        "error": (
            f"ဗီဒီယိုဖိုင်ကြီးလွန်းနေပါသည်။ "
            f"အများဆုံး {max_mb} MB အထိသာ တင်နိုင်ပါသည်။"
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
    app.logger.exception("Unhandled server error")

    return jsonify({
        "error": (
            "Server ဘက်မှာ ပြဿနာဖြစ်နေပါသည်။ "
            "Render Logs ကို စစ်ပါ။"
        )
    }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
