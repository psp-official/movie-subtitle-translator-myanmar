
import os
import re
import json
import tempfile
import subprocess
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
    milliseconds = int(round(max(0, seconds) * 1000))

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
    text = text.strip()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE
    )
    return text.strip()


def translate_batch(model, segments):
    """Translate a group of Chinese dialogue lines into Burmese."""
    payload = [
        {"id": i, "text": segment["text"]}
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
- If a line is unclear, translate conservatively.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = model.generate_content(prompt)
    result = json.loads(clean_json_response(response.text))

    translations = {
        int(item["id"]): str(item["myanmar"]).strip()
        for item in result
        if "id" in item and "myanmar" in item
    }

    return [
        translations.get(i, segment["text"])
        for i, segment in enumerate(segments)
    ]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/process")
def process_video():
    if "video" not in request.files:
        return jsonify(error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"), 400

    video = request.files["video"]

    if not video.filename:
        return jsonify(error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"), 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(
            error="MP4, MOV, MKV သို့မဟုတ် WEBM ဖိုင်ကို ရွေးပါ။"
        ), 400

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not groq_key or not gemini_key:
        return jsonify(
            error=(
                "Server Environment Variables ထဲမှာ "
                "GROQ_API_KEY နဲ့ GEMINI_API_KEY ထည့်ပေးပါ။"
            )
        ), 500

    try:
        from groq import Groq
        import google.generativeai as genai
    except ImportError:
        return jsonify(
            error="လိုအပ်သော Python libraries များ မတပ်ဆင်ရသေးပါ။"
        ), 500

    job_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
    video_path = job_dir / filename
    audio_path = job_dir / "audio.mp3"

    try:
        video.save(video_path)

        # Extract audio without modifying the original video.
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
            return jsonify(
                error="ဗီဒီယိုထဲက အသံကို ထုတ်ယူလို့မရပါ။"
            ), 400

        # Speech-to-text using Groq Whisper.
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

        for segment in (transcription.segments or []):
            text = str(segment.text).strip()
            start = float(segment.start)
            end = float(segment.end)

            if text:
                raw_segments.append({
                    "start": start,
                    "end": max(start + 0.15, end),
                    "text": text
                })

        if not raw_segments:
            return jsonify(
                error=(
                    "စကားပြောကို မတွေ့ပါ။ "
                    "အသံပါသော တရုတ်ဗီဒီယိုနဲ့ ပြန်စမ်းပါ။"
                )
            ), 422

        # Translate dialogue into Burmese using Gemini.
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        translated_lines = []

        batch_size = 25

        for start_index in range(0, len(raw_segments), batch_size):
            batch = raw_segments[
                start_index:start_index + batch_size
            ]

            try:
                translations = translate_batch(model, batch)
            except Exception:
                # Keep original text for failed batches so the
                # user can still review and edit those lines.
                translations = [
                    item["text"] for item in batch
                ]

            translated_lines.extend(translations)

        # Build subtitle editor data.
        items = []

        for index, segment in enumerate(raw_segments):
            items.append({
                "index": index + 1,
                "time": (
                    f"{make_timestamp(segment['start'])} --> "
                    f"{make_timestamp(segment['end'])}"
                ),
                "text": translated_lines[index],
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
        return jsonify(
            error="ဗီဒီယိုအသံထုတ်ယူချိန် ကျော်လွန်သွားပါတယ်။"
        ), 504

    except Exception:
        app.logger.exception("Video processing failed")
        return jsonify(
            error=(
                "လုပ်ဆောင်မှု မအောင်မြင်ပါ။ "
                "API Key၊ API quota နဲ့ Server Logs ကို စစ်ပါ။"
            )
        ), 500

    finally:
        # Clean up temporary files after processing.
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/api/export")
def export_srt():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify(error="Subtitle data မမှန်ပါ။"), 400

    output = []

    for item in items:
        if not isinstance(item, dict):
            continue

        timestamp = str(item.get("time", "")).strip()
        text = str(item.get("text", "")).strip()

        if "-->" not in timestamp or not text:
            continue

        # Prevent invalid multiline SRT structure.
        output.append(
            f"{len(output) + 1}\n{timestamp}\n{text}"
        )

    srt_content = "\n\n".join(output) + (
        "\n" if output else ""
    )

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
