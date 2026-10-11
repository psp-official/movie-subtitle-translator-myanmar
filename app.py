import os
import re
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Maximum video upload size: 1 GB
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

UPLOAD_DIR = Path(tempfile.gettempdir()) / "movie_subtitle_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def make_timestamp(seconds):
    """Convert seconds into SRT timestamp format."""
    milliseconds = int(round(max(0, float(seconds)) * 1000))

    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)

    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{seconds:02},"
        f"{milliseconds:03}"
    )


def clean_json_response(text):
    """Remove Markdown code fences from AI responses."""
    text = str(text).strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def get_field(obj, field, default=None):
    """Read a field from either a dictionary or an object."""
    if isinstance(obj, dict):
        return obj.get(field, default)

    return getattr(obj, field, default)


def translate_batch(model, segments):
    """Translate Chinese dialogue into natural conversational Burmese."""

    payload = [
        {
            "id": i,
            "text": segment["text"]
        }
        for i, segment in enumerate(segments)
    ]

    prompt = f"""
Translate the following Chinese movie dialogue into natural,
conversational Burmese (Myanmar language).

Rules:
- Preserve the original meaning, emotion, and character intent.
- Use natural spoken Burmese suitable for movie subtitles.
- Do not summarize or add narration.
- Do not explain the translation.
- Keep every ID and preserve the original order.
- Return ONLY a valid JSON array.
- Each item must contain "id" and "myanmar".
- Translate every line.
- If a line is unclear, translate conservatively.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = model.generate_content(prompt)

    response_text = get_field(response, "text", "")

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(clean_json_response(response_text))

    if not isinstance(result, list):
        raise ValueError("Gemini response is not a JSON array.")

    translations = {}

    for item in result:
        if not isinstance(item, dict):
            continue

        if "id" not in item or "myanmar" not in item:
            continue

        try:
            translations[int(item["id"])] = str(
                item["myanmar"]
            ).strip()
        except (TypeError, ValueError):
            continue

    return [
        translations.get(i, segment["text"])
        for i, segment in enumerate(segments)
    ]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/process")
def process_video():

    # Check uploaded file
    if "video" not in request.files:
        return jsonify(
            error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"
        ), 400

    video = request.files["video"]

    if not video.filename:
        return jsonify(
            error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"
        ), 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(
            error="MP4, MOV, MKV သို့မဟုတ် WEBM ဖိုင်ကို ရွေးပါ။"
        ), 400

    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not groq_key or not gemini_key:
        return jsonify(
            error=(
                "Render Environment Variables ထဲမှာ "
                "GROQ_API_KEY နဲ့ GEMINI_API_KEY "
                "နှစ်ခုလုံး ထည့်ထားကြောင်း စစ်ပါ။"
            )
        ), 500

    # Import API libraries
    try:
        from groq import Groq
        import google.generativeai as genai

    except ImportError:
        app.logger.exception("Required library import failed")

        return jsonify(
            error="လိုအပ်သော Python libraries များ မတပ်ဆင်ရသေးပါ။"
        ), 500

    # Create temporary working directory
    job_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))

    video_path = job_dir / filename
    audio_path = job_dir / "audio.mp3"

    try:
        video.save(video_path)

        # Extract audio using FFmpeg.
        # The original video is not modified or re-rendered.
        command = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            str(audio_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode != 0 or not audio_path.exists():
            app.logger.error(
                "FFmpeg audio extraction failed: %s",
                result.stderr[-3000:]
            )

            return jsonify(
                error="ဗီဒီယိုထဲက အသံကို ထုတ်ယူလို့မရပါ။"
            ), 400

        if audio_path.stat().st_size == 0:
            return jsonify(
                error="ဗီဒီယိုထဲမှာ အသံမတွေ့ပါ။"
            ), 400

        # Speech-to-text using Groq Whisper
        groq_client = Groq(api_key=groq_key)

        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model="whisper-large-v3",
                language="zh",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        raw_segments = []

        # Support both dictionaries and objects returned by the API.
        transcription_segments = get_field(
            transcription,
            "segments",
            []
        ) or []

        for segment in transcription_segments:
            try:
                text = str(
                    get_field(segment, "text", "") or ""
                ).strip()

                start = float(
                    get_field(segment, "start", 0) or 0
                )

                end = float(
                    get_field(segment, "end", start) or start
                )

            except (TypeError, ValueError):
                app.logger.warning(
                    "Skipping invalid transcription segment: %r",
                    segment
                )
                continue

            if not text:
                continue

            start = max(0.0, start)
            end = max(start + 0.15, end)

            raw_segments.append({
                "start": start,
                "end": end,
                "text": text
            })

        if not raw_segments:
            return jsonify(
                error=(
                    "စကားပြောစာသား မတွေ့ပါ။ "
                    "အသံပါသော တရုတ်ဗီဒီယိုနဲ့ ပြန်စမ်းပါ။"
                )
            ), 422

        # Configure Gemini for Burmese translation
        genai.configure(api_key=gemini_key)

        model = genai.GenerativeModel("gemini-3.8-flash")

        translated_lines = []

        # Translate in batches to avoid oversized requests
        batch_size = 25

        for start_index in range(
            0,
            len(raw_segments),
            batch_size
        ):
            batch = raw_segments[
                start_index:start_index + batch_size
            ]

            try:
                translations = translate_batch(model, batch)

            except Exception:
                app.logger.exception(
                    "Gemini translation batch failed at index %s",
                    start_index
                )

                # Keep the original Chinese text for failed batches.
                # The user can review or edit these lines.
                translations = [
                    item["text"] for item in batch
                ]

            translated_lines.extend(translations)

        # Build subtitle editor data
        items = []

        for index, segment in enumerate(raw_segments):
            translated_text = (
                translated_lines[index]
                if index < len(translated_lines)
                else segment["text"]
            )

            items.append({
                "index": index + 1,
                "time": (
                    f"{make_timestamp(segment['start'])} --> "
                    f"{make_timestamp(segment['end'])}"
                ),
                "text": translated_text,
                "source": segment["text"]
            })

        return jsonify(
            items=items,
            notice=(
                "အသံမှ စကားပြောကို ဖမ်းယူပြီး ဘာသာပြန်ထားပါပြီ။ "
                "SRT မဒေါင်းလုဒ်ခင် စာတန်းများကို ပြန်စစ်ပါ။ "
                "ဗီဒီယိုပေါ်က တရုတ်စာတန်း OCR ကို "
                "ဒီဗားရှင်းမှာ မထည့်ရသေးပါ။"
            )
        )

    except subprocess.TimeoutExpired:
        app.logger.exception("FFmpeg processing timed out")

        return jsonify(
            error="ဗီဒီယိုအသံထုတ်ယူချိန် ကျော်လွန်သွားပါတယ်။"
        ), 504

    except Exception:
        app.logger.exception("Video processing failed")

        return jsonify(
            error=(
                "လုပ်ဆောင်မှု မအောင်မြင်ပါ။ "
                "Render Logs၊ API Key နဲ့ API quota ကို စစ်ပါ။"
            )
        ), 500

    finally:
        # Delete temporary video/audio files after processing
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/api/export")
def export_srt():

    data = request.get_json(silent=True) or {}
    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify(
            error="Subtitle data မမှန်ပါ။"
        ), 400

    output = []

    for item in items:
        if not isinstance(item, dict):
            continue

        timestamp = str(item.get("time", "")).strip()
        text = str(item.get("text", "")).strip()

        if "-->" not in timestamp or not text:
            continue

        # Prevent malformed SRT blocks
        output.append(
            f"{len(output) + 1}\n"
            f"{timestamp}\n"
            f"{text}"
        )

    srt_content = "\n\n".join(output)

    if srt_content:
        srt_content += "\n"

    return send_file(
        BytesIO(srt_content.encode("utf-8-sig")),
        mimetype="application/x-subrip",
        as_attachment=True,
        download_name="myanmar_subtitles.srt"
    )


@app.errorhandler(413)
def file_too_large(error):
    return jsonify(
        error="ဖိုင်အရွယ်အစား အရမ်းကြီးနေပါတယ်။"
    ), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
