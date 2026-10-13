
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import OrderedDict
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = Path(tempfile.gettempdir()) / "movie_subtitle_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
GEMINI_BATCH_SIZE = max(1, min(30, int(os.getenv("GEMINI_BATCH_SIZE", "20"))))
GEMINI_RETRIES = max(0, min(5, int(os.getenv("GEMINI_RETRIES", "3"))))
OCR_FPS = max(1, min(8, int(os.getenv("OCR_FPS", "4"))))
OCR_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("OCR_CONFIDENCE", "0.45"))))
SUBTITLE_DELAY_SECONDS = float(os.getenv("SUBTITLE_DELAY_SECONDS", "0.0"))
AUDIO_CHUNK_SECONDS = max(60, min(600, int(os.getenv("AUDIO_CHUNK_SECONDS", "480"))))

TRANSLATION_CACHE_MAX = 5000
TRANSLATION_CACHE = OrderedDict()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def make_timestamp(seconds):
    milliseconds = int(round(max(0.0, float(seconds)) * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def format_subtitle(text, max_chars=22):
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    if " " not in text:
        if len(text) <= max_chars * 2:
            return text[:max_chars] + "\n" + text[max_chars:]
        return text[:max_chars] + "\n" + text[max_chars:max_chars * 2]

    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        elif not current:
            lines.append(word[:max_chars])
            current = word[max_chars:]
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    if len(lines) <= 2:
        return "\n".join(lines)

    return lines[0] + "\n" + " ".join(lines[1:])


def clean_json_response(text):
    text = str(text or "").strip()
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def cache_get(text):
    key = re.sub(r"\s+", " ", str(text or "")).strip()
    value = TRANSLATION_CACHE.get(key)

    if value is not None:
        TRANSLATION_CACHE.move_to_end(key)

    return key, value


def cache_put(key, value):
    if not str(value or "").strip():
        return

    TRANSLATION_CACHE[key] = value
    TRANSLATION_CACHE.move_to_end(key)

    while len(TRANSLATION_CACHE) > TRANSLATION_CACHE_MAX:
        TRANSLATION_CACHE.popitem(last=False)


def make_translation_prompt(uncached):
    return f"""
You are an expert Chinese-to-Myanmar subtitle translator.

Translate every Chinese dialogue line into natural, concise,
spoken Myanmar Burmese suitable for movie subtitles.

RULES:
- Preserve meaning, emotion, intent, and character relationships.
- Use natural Burmese dialogue, not stiff word-for-word translation.
- Interpret idioms, insults, jokes, honorifics, and fantasy terms from context.
- Do not add explanations, narration, labels, or translator notes.
- Use Myanmar script.
- Translate every item. Do not omit, merge, reorder, or duplicate IDs.
- Keep dialogue concise enough for subtitles.
- Return ONLY a valid JSON array.
- Each object must have integer "id" and string "myanmar".

Example:
[{{"id": 0, "myanmar": "မင်း ဘာလုပ်နေတာလဲ။"}}]

Input:
{json.dumps(uncached, ensure_ascii=False)}
""".strip()


def translate_batch(client, segments):
    translations = {}
    uncached = []
    cache_keys = {}

    for index, segment in enumerate(segments):
        source_text = str(segment.get("text", "")).strip()
        key, cached = cache_get(source_text)
        cache_keys[index] = key

        if cached is not None:
            translations[index] = cached
        else:
            uncached.append({
                "id": index,
                "text": source_text
            })

    if uncached:
        from google.genai import types

        prompt = make_translation_prompt(uncached)
        response_text = ""

        for attempt in range(GEMINI_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )

                response_text = str(
                    get_field(response, "text", "") or ""
                ).strip()

                if not response_text:
                    raise ValueError("Gemini returned an empty response.")

                break

            except Exception as exc:
                if attempt >= GEMINI_RETRIES:
                    app.logger.exception("Gemini translation failed")
                    raise RuntimeError(
                        f"Gemini translation failed: {exc}"
                    ) from exc

                time.sleep(min(12, 2 ** (attempt + 1)))

        try:
            parsed = json.loads(clean_json_response(response_text))

        except json.JSONDecodeError:
            app.logger.warning(
                "Gemini returned invalid JSON; attempting repair."
            )

            repair_prompt = (
                "Convert the following content into ONLY a valid JSON "
                'array with integer "id" and string "myanmar". '
                "Preserve the translation text. Fix JSON syntax only.\n\n"
                + response_text
            )

            repaired = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=repair_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )

            parsed = json.loads(
                clean_json_response(
                    get_field(repaired, "text", "")
                )
            )

        if isinstance(parsed, dict):
            parsed = parsed.get(
                "translations",
                parsed.get("items", [])
            )

        if not isinstance(parsed, list):
            raise RuntimeError("Gemini response was not a JSON array.")

        returned = {}

        for item in parsed:
            if not isinstance(item, dict):
                continue

            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue

            translated = str(
                item.get("myanmar", "") or ""
            ).strip()

            if translated:
                returned[item_id] = translated

        for item in uncached:
            index = item["id"]
            translated = returned.get(index, "")

            if translated:
                translations[index] = translated
                cache_put(cache_keys[index], translated)
            else:
                # Avoid returning an empty subtitle.
                translations[index] = item["text"]

    return [
        str(translations.get(i, segments[i].get("text", ""))).strip()
        for i in range(len(segments))
    ]



def run_ocr_extraction(video_path):
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video for OCR.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_interval = max(1, int(round(fps / OCR_FPS)))
    sample_interval = frame_interval / fps

    segments = []
    current_text = ""
    current_start = 0.0
    last_seen = 0.0
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_count % frame_interval == 0:
                height, width = frame.shape[:2]

                # Include the lower 60% of the frame to support
                # both horizontal and vertical videos.
                crop = frame[int(height * 0.40):height, 0:width]

                if crop.size:
                    result, _ = ocr(crop)
                else:
                    result = None

                lines = []

                if result:
                    for line in result:
                        try:
                            text = str(line[1]).strip()
                            confidence = float(line[2])
                        except (IndexError, TypeError, ValueError):
                            continue

                        if (
                            confidence >= OCR_CONFIDENCE
                            and re.search(r"[\u4e00-\u9fff]", text)
                        ):
                            lines.append(text)

                frame_text = re.sub(
                    r"\s+", " ", " ".join(lines)
                ).strip()

                timestamp = frame_count / fps

                if frame_text:
                    if not current_text:
                        current_text = frame_text
                        current_start = timestamp
                        last_seen = timestamp

                    else:
                        similarity = SequenceMatcher(
                            None, current_text, frame_text
                        ).ratio()

                        if similarity >= 0.72:
                            # Reduce unnecessary splitting when OCR
                            # makes small character-recognition errors.
                            if len(frame_text) >= len(current_text) * 0.7:
                                current_text = frame_text

                            last_seen = timestamp

                        else:
                            end_time = max(
                                current_start + 0.15,
                                last_seen + sample_interval
                            )

                            segments.append({
                                "start": current_start,
                                "end": end_time,
                                "text": current_text
                            })

                            current_text = frame_text
                            current_start = timestamp
                            last_seen = timestamp

                elif current_text:
                    end_time = max(
                        current_start + 0.15,
                        last_seen + sample_interval
                    )

                    segments.append({
                        "start": current_start,
                        "end": end_time,
                        "text": current_text
                    })

                    current_text = ""

            frame_count += 1

        if current_text:
            end_time = max(
                current_start + 0.15,
                last_seen + sample_interval
            )

            segments.append({
                "start": current_start,
                "end": end_time,
                "text": current_text
            })

    finally:
        cap.release()

    final_segments = []

    for segment in segments:
        segment["start"] = max(0.0, float(segment["start"]))
        segment["end"] = max(
            segment["start"] + 0.15,
            float(segment["end"])
        )
        segment["text"] = re.sub(
            r"\s+", " ", segment["text"]
        ).strip()

        if not segment["text"]:
            continue

        if segment["end"] - segment["start"] < 0.20:
            continue

        if final_segments:
            previous = final_segments[-1]

            same_text = (
                SequenceMatcher(
                    None,
                    previous["text"],
                    segment["text"]
                ).ratio() >= 0.90
            )

            adjacent = (
                segment["start"] - previous["end"] <= 0.12
            )

            if same_text and adjacent:
                previous["end"] = segment["end"]
                continue

        final_segments.append(segment)

    return final_segments


def check_has_hardcoded_subs(video_path):
    """Quickly scan up to 60 seconds for Chinese hardcoded subtitles."""
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25.0

        frame_interval = max(1, int(round(fps)))
        max_scan_frames = int(fps * 60)

        ocr = RapidOCR()
        chinese_frames = 0
        frame_count = 0

        while frame_count < max_scan_frames:
            success, frame = cap.read()

            if not success:
                break

            if frame_count % frame_interval == 0:
                height, width = frame.shape[:2]
                crop = frame[int(height * 0.40):height, 0:width]

                result, _ = ocr(crop)

                if result:
                    found_chinese = False

                    for line in result:
                        try:
                            text = str(line[1])
                            confidence = float(line[2])
                        except (IndexError, TypeError, ValueError):
                            continue

                        if (
                            confidence >= OCR_CONFIDENCE
                            and re.search(r"[\u4e00-\u9fff]", text)
                        ):
                            found_chinese = True
                            break

                    if found_chinese:
                        chinese_frames += 1

                        if chinese_frames >= 3:
                            return True

            frame_count += 1

    except Exception:
        app.logger.exception("OCR subtitle detection failed")

    finally:
        cap.release()

    return False


def run_ffmpeg(args, timeout=1800):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y"
    ] + args

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        message = (result.stderr or "ffmpeg failed").strip()
        raise RuntimeError(message[-1200:])

    return result


def transcribe_audio(video_path, job_dir, groq_client, total_delay=0.0):
    """Extract audio, split it into chunks, and transcribe each chunk."""
    audio_path = job_dir / "audio.mp3"

    run_ffmpeg([
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        str(audio_path)
    ])

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise ValueError("ဗီဒီယိုထဲမှာ အသံမတွေ့ပါ။")

    chunk_dir = job_dir / "audio_chunks"
    chunk_dir.mkdir(exist_ok=True)

    chunk_pattern = str(chunk_dir / "chunk_%04d.mp3")

    run_ffmpeg([
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(AUDIO_CHUNK_SECONDS),
        "-reset_timestamps", "1",
        "-c", "copy",
        chunk_pattern
    ])

    chunk_files = sorted(chunk_dir.glob("chunk_*.mp3"))

    if not chunk_files:
        chunk_files = [audio_path]

    raw_segments = []
    offset = 0.0

    for chunk_path in chunk_files:
        try:
            transcription = (
                groq_client.audio.transcriptions.create(
                    file=(
                        chunk_path.name,
                        chunk_path.read_bytes()
                    ),
                    model=GROQ_STT_MODEL,
                    language="zh",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            )

        except Exception as exc:
            app.logger.exception(
                "Groq transcription failed: %s",
                chunk_path.name
            )

            raise RuntimeError(
                "အသံကို စာသားပြောင်းမရပါ။ GROQ_API_KEY၊ "
                "API quota နဲ့ STT model ကို စစ်ပါ။"
            ) from exc

        chunk_segments = (
            get_field(transcription, "segments", []) or []
        )

        for segment in chunk_segments:
            text = str(
                get_field(segment, "text", "") or ""
            ).strip()

            if not text:
                continue

            try:
                start = (
                    float(get_field(segment, "start", 0.0))
                    + offset
                )

                end = (
                    float(get_field(segment, "end", 0.0))
                    + offset
                )

            except (TypeError, ValueError):
                continue

            if end <= start:
                end = start + 0.25

            raw_segments.append({
                "start": max(0.0, start + total_delay),
                "end": max(0.0, end + total_delay),
                "text": text
            })

        # Use the actual chunk duration to calculate the next offset.
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(chunk_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            duration = (
                float(probe.stdout.strip())
                if probe.returncode == 0
                else AUDIO_CHUNK_SECONDS
            )

        except (ValueError, subprocess.SubprocessError):
            duration = AUDIO_CHUNK_SECONDS

        offset += duration

    raw_segments.sort(
        key=lambda item: (item["start"], item["end"])
    )

    cleaned = []

    for segment in raw_segments:
        if segment["end"] <= 0:
            continue

        segment["start"] = max(0.0, segment["start"])
        segment["end"] = max(
            segment["start"] + 0.15,
            segment["end"]
        )

        if cleaned and segment["start"] < cleaned[-1]["end"]:
            cleaned[-1]["end"] = max(
                cleaned[-1]["start"] + 0.15,
                segment["start"]
            )

        cleaned.append(segment)

    return [
        segment for segment in cleaned
        if segment["end"] > segment["start"]
    ]



@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/process")
def process_video():
    if "video" not in request.files:
        return jsonify(
            error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"
        ), 400

    video = request.files["video"]

    if not video.filename:
        return jsonify(
            error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"
        ), 400

    safe_name = secure_filename(video.filename)

    if (
        not safe_name
        or Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS
    ):
        return jsonify(
            error="MP4, MOV, MKV သို့မဟုတ် WEBM ဖိုင်ကို ရွေးပါ။"
        ), 400

    try:
        user_delay = float(
            request.form.get("delay", "0.0")
        )
    except (TypeError, ValueError):
        user_delay = 0.0

    user_delay = max(-600.0, min(600.0, user_delay))

    mode = str(
        request.form.get("mode", "auto")
    ).strip().lower()

    if mode not in {"auto", "ocr", "audio"}:
        mode = "auto"

    gemini_key = os.getenv(
        "GEMINI_API_KEY", ""
    ).strip()

    groq_key = os.getenv(
        "GROQ_API_KEY", ""
    ).strip()

    if not gemini_key:
        return jsonify(
            error="GEMINI_API_KEY ကို .env ထဲတွင် ထည့်ပါ။"
        ), 500

    if mode == "audio" and not groq_key:
        return jsonify(
            error="Audio Mode အတွက် GROQ_API_KEY ကို .env ထဲတွင် ထည့်ပါ။"
        ), 500

    try:
        from google import genai
    except ImportError:
        return jsonify(
            error=(
                "google-genai library မတပ်ဆင်ရသေးပါ။ "
                "pip install google-genai"
            )
        ), 500

    if mode == "audio":
        try:
            from groq import Groq
        except ImportError:
            return jsonify(
                error="groq library မတပ်ဆင်ရသေးပါ။ pip install groq"
            ), 500

    job_dir = Path(
        tempfile.mkdtemp(
            prefix="movie_subtitle_",
            dir=UPLOAD_DIR
        )
    )

    video_path = job_dir / safe_name

    try:
        video.save(video_path)

        if (
            not video_path.exists()
            or video_path.stat().st_size == 0
        ):
            return jsonify(
                error="ဗီဒီယိုဖိုင်ကို သိမ်းဆည်းမရပါ။"
            ), 400

        gemini_client = genai.Client(
            api_key=gemini_key
        )

        raw_segments = []
        source_type = ""

        # Decide whether OCR should be used.
        use_ocr = mode == "ocr"

        if mode == "auto":
            try:
                use_ocr = check_has_hardcoded_subs(
                    video_path
                )
            except Exception:
                app.logger.exception(
                    "OCR subtitle detection failed"
                )
                use_ocr = False

        # Read hardcoded subtitles from video frames.
        if use_ocr:
            try:
                raw_segments = run_ocr_extraction(
                    video_path
                )

                if raw_segments:
                    source_type = (
                        "OCR ဖြင့် ဗီဒီယိုပေါ်ရှိ စာတန်းကို "
                        "ဖတ်ယူထားပါသည်။"
                    )

            except Exception:
                app.logger.exception(
                    "OCR extraction failed"
                )
                raw_segments = []

        # Fall back to speech recognition if OCR finds no usable text.
        if not raw_segments:
            if not groq_key:
                return jsonify(
                    error=(
                        "ဗီဒီယိုပေါ်ရှိ တရုတ်စာတန်းကို "
                        "မဖတ်နိုင်ပါ။ အသံမှ စာသားထုတ်ယူရန် "
                        "GROQ_API_KEY ကို .env ထဲတွင် ထည့်ပါ။"
                    )
                ), 422

            try:
                from groq import Groq
            except ImportError:
                return jsonify(
                    error=(
                        "groq library မတပ်ဆင်ရသေးပါ။ "
                        "pip install groq"
                    )
                ), 500

            groq_client = Groq(
                api_key=groq_key
            )

            total_delay = (
                SUBTITLE_DELAY_SECONDS + user_delay
            )

            raw_segments = transcribe_audio(
                video_path,
                job_dir,
                groq_client,
                total_delay=total_delay
            )

            source_type = (
                "Whisper အသံမှတ်သားစနစ်ဖြင့် "
                "စာသားထုတ်ယူထားပါသည်။"
            )

        else:
            # OCR timestamps also respect the user's sync adjustment.
            total_delay = (
                SUBTITLE_DELAY_SECONDS + user_delay
            )

            for segment in raw_segments:
                original_start = float(
                    segment["start"]
                )

                original_end = float(
                    segment["end"]
                )

                segment["start"] = max(
                    0.0,
                    original_start + total_delay
                )

                segment["end"] = max(
                    segment["start"] + 0.15,
                    original_end + total_delay
                )

        if not raw_segments:
            return jsonify(
                error=(
                    "ဖတ်ယူနိုင်သော တရုတ်စာတန်း သို့မဟုတ် "
                    "စကားပြောစာသား မတွေ့ပါ။"
                )
            ), 422

        raw_segments.sort(
            key=lambda item: (
                item["start"],
                item["end"]
            )
        )

        # Translate in batches to reduce API calls.
        translated_lines = []

        for start_index in range(
            0,
            len(raw_segments),
            GEMINI_BATCH_SIZE
        ):
            batch = raw_segments[
                start_index:
                start_index + GEMINI_BATCH_SIZE
            ]

            try:
                translated_lines.extend(
                    translate_batch(
                        gemini_client,
                        batch
                    )
                )

            except Exception as exc:
                app.logger.exception(
                    "Translation failed"
                )

                return jsonify(
                    error=(
                        "မြန်မာဘာသာပြန်ခြင်း မအောင်မြင်ပါ။ "
                        "GEMINI_API_KEY၊ model၊ quota နဲ့ "
                        "internet connection ကို စစ်ပါ။"
                    ),
                    detail=str(exc)[:500]
                ), 502

        items = []

        for index, segment in enumerate(raw_segments):
            if index < len(translated_lines):
                translated = translated_lines[index]
            else:
                translated = segment["text"]

            items.append({
                "index": index + 1,
                "time": (
                    f"{make_timestamp(segment['start'])} --> "
                    f"{make_timestamp(segment['end'])}"
                ),
                "text": format_subtitle(
                    translated
                ),
                "source": segment["text"]
            })

        return jsonify(
            items=items,
            notice=(
                f"မြန်မာဘာသာပြန်ပြီးပါပြီ။ {source_type} "
                "စာတန်းများကို ပြန်စစ်ပြီး SRT ထုတ်ပါ။"
            )
        )

    except subprocess.TimeoutExpired:
        app.logger.exception(
            "Media processing timed out"
        )

        return jsonify(
            error=(
                "ဗီဒီယိုပြင်ဆင်ချိန် ကြာလွန်းသွားပါသည်။ "
                "ဖိုင်တိုတစ်ခုဖြင့် ပြန်စမ်းပါ။"
            )
        ), 504

    except ValueError as exc:
        return jsonify(
            error=str(exc)
        ), 400

    except Exception:
        app.logger.error(
            "Global processing error:\n%s",
            traceback.format_exc()
        )

        return jsonify(
            error=(
                "စနစ်ချို့ယွင်းမှု ဖြစ်ပေါ်သွားပါသည်။ "
                "Server console log ကို စစ်ဆေးပါ။"
            )
        ), 500

    finally:
        shutil.rmtree(
            job_dir,
            ignore_errors=True
        )


@app.post("/api/export")
def export_srt():
    payload = request.get_json(
        silent=True
    ) or {}

    items = payload.get(
        "items", []
    )

    if not isinstance(items, list):
        return jsonify(
            error="SRT export အတွက် items ပုံစံ မမှန်ပါ။"
        ), 400

    output = []

    for item in items:
        if not isinstance(item, dict):
            continue

        timestamp = str(
            item.get("time", "")
        ).strip()

        text = str(
            item.get("text", "")
        ).strip()

        if "-->" in timestamp and text:
            output.append(
                f"{len(output) + 1}\n{timestamp}\n{text}"
            )

    if not output:
        return jsonify(
            error="ထုတ်ယူရန် စာတန်းမရှိပါ။"
        ), 400

    data = (
        "\n\n".join(output) + "\n"
    ).encode("utf-8-sig")

    return send_file(
        BytesIO(data),
        mimetype="application/x-subrip",
        as_attachment=True,
        download_name="myanmar_subtitles.srt"
    )


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(
        error=(
            "ဖိုင်အရွယ်အစား ကြီးလွန်းပါသည်။ "
            "MAX_UPLOAD_MB ကို စစ်ဆေးပါ။"
        )
    ), 413


@app.errorhandler(500)
def server_error(_error):
    return jsonify(
        error=(
            "Internal Server Error: "
            "Backend တွင် ပြဿနာရှိနေပါသည်။"
        )
    ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "5000")
        ),
        debug=False
    )

