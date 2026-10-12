import os
import re
import json
import tempfile
import subprocess
import shutil
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
UPLOAD_DIR = Path(tempfile.gettempdir()) / "movie_subtitle_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BATCH_SIZE = max(1, min(40, int(os.getenv("GEMINI_BATCH_SIZE", "25"))))
GEMINI_RETRIES = max(0, min(5, int(os.getenv("GEMINI_RETRIES", "3"))))
SUBTITLE_DELAY_SECONDS = float(os.getenv("SUBTITLE_DELAY_SECONDS", "0.0"))
TRANSLATION_CACHE_MAX = 5000
TRANSLATION_CACHE = OrderedDict()

def get_field(obj, name, default=None):
    if isinstance(obj, dict): return obj.get(name, default)
    return getattr(obj, name, default)

def make_timestamp(seconds):
    milliseconds = int(round(max(0.0, float(seconds)) * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_subtitle(text, max_chars=22):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text: return ""
    if len(text) <= max_chars: return text
    words = text.split(" ")
    if len(words) == 1:
        parts = re.findall(r".{1,22}", text)
        return "\n".join(parts[:2]) if len(parts) <= 2 else (parts[0] + "\n" + "".join(parts[1:]))
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current: current = candidate
        else:
            lines.append(current)
            current = word
    if current: lines.append(current)
    if len(lines) <= 2: return "\n".join(lines)
    return lines[0] + "\n" + " ".join(lines[1:])

def clean_json_response(text):
    text = str(text or "").strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

def _cache_get(text):
    key = re.sub(r"\s+", " ", str(text or "")).strip()
    value = TRANSLATION_CACHE.get(key)
    if value is not None: TRANSLATION_CACHE.move_to_end(key)
    return key, value

def _cache_put(key, value):
    TRANSLATION_CACHE[key] = value
    TRANSLATION_CACHE.move_to_end(key)
    while len(TRANSLATION_CACHE) > TRANSLATION_CACHE_MAX: TRANSLATION_CACHE.popitem(last=False)

def translate_batch(model, segments):
    translations = {}
    uncached = []
    cache_keys = {}
    for idx, segment in enumerate(segments):
        key, cached = _cache_get(segment["text"])
        cache_keys[idx] = key
        if cached is not None: translations[idx] = cached
        else: uncached.append({"id": idx, "text": segment["text"]})

    if uncached:
        prompt = f"""You are a professional Chinese-to-Myanmar subtitle translator.
Translate each Chinese dialogue line into natural spoken Burmese.
RULES: Output Burmese only. Preserve meaning, emotion. No narration. Keep concise. Return ONLY valid JSON array with integer "id" and string "myanmar".
Example: [{{"id": 0, "myanmar": "မင်း ဘာလုပ်နေတာလဲ။"}}]
Input: {json.dumps(uncached, ensure_ascii=False)}"""
        response = None
        for attempt in range(GEMINI_RETRIES + 1):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as exc:
                if attempt >= GEMINI_RETRIES: raise
                time.sleep(min(30, 2 ** attempt * 2))

        response_text = get_field(response, "text", "")
        if not response_text: raise ValueError("Empty response.")
        result = json.loads(clean_json_response(response_text))
        
        returned = {}
        for item in result:
            try:
                if str(item.get("myanmar", "")).strip(): returned[int(item["id"])] = str(item["myanmar"]).strip()
            except (TypeError, ValueError, KeyError): continue

        for item in uncached:
            idx = item["id"]
            translations[idx] = returned.get(idx, "")
            _cache_put(cache_keys[idx], returned.get(idx, ""))

    return [translations[i] for i in range(len(segments))]

def get_ocr_instance():
    from paddleocr import PaddleOCR
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    # Disable IR optimization and MKLDNN to prevent AVX/SIGILL crashes on Render CPUs
    return PaddleOCR(
        use_angle_cls=False, 
        lang='ch', 
        use_gpu=False, 
        show_log=False,
        ir_optim=False,
        enable_mkldnn=False,
        use_tensorrt=False
    )

def check_has_hardcoded_subs(video_path):
    import cv2
    try:
        ocr = get_ocr_instance()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        frame_interval = int(max(1, round(fps))) # 1 fps
        max_scan_frames = int(fps * 60) # Scan up to 60 sec
        
        frame_count = 0
        chinese_frames = 0
        success = True
        
        while success and frame_count < max_scan_frames:
            success, frame = cap.read()
            if not success: break
            
            if frame_count % frame_interval == 0:
                h, w, _ = frame.shape
                crop = frame[int(h*0.75):h, :]
                result = ocr.ocr(crop, cls=False)
                
                if result and result[0]:
                    text = "".join([line[1][0] for line in result[0] if line[1][1] > 0.6])
                    if re.search(r'[\u4e00-\u9fff]', text):
                        chinese_frames += 1
                        if chinese_frames >= 5:
                            cap.release()
                            return True
            frame_count += 1
        cap.release()
    except Exception as e:
        app.logger.warning(f"OCR Check failed: {e}")
    return False

def run_ocr_extraction(video_path):
    import cv2
    from difflib import SequenceMatcher
    
    ocr = get_ocr_instance()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    
    frame_interval = int(max(1, round(fps / 2))) # 2 fps
    
    segments = []
    current_text = ""
    current_start = 0.0
    
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success: break
        
        if frame_count % frame_interval == 0:
            h, w, _ = frame.shape
            crop = frame[int(h*0.75):h, :]
            result = ocr.ocr(crop, cls=False)
            
            frame_text = ""
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line[1][1] > 0.6]
                frame_text = " ".join(lines).strip()
            
            timestamp = frame_count / fps
            similarity = SequenceMatcher(None, current_text, frame_text).ratio()
            
            if similarity < 0.7 or not frame_text:
                if current_text and re.search(r'[\u4e00-\u9fff]', current_text):
                    segments.append({"start": current_start, "end": timestamp, "text": current_text})
                current_text = frame_text
                current_start = timestamp
                
        frame_count += 1
        
    if current_text and re.search(r'[\u4e00-\u9fff]', current_text):
        segments.append({"start": current_start, "end": frame_count / fps, "text": current_text})
        
    cap.release()
    
    final_segments = [s for s in segments if (s["end"] - s["start"]) > 0.3]
    return final_segments


@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/process")
def process_video():
    if "video" not in request.files: return jsonify(error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"), 400
    video = request.files["video"]
    
    try: user_delay = float(request.form.get("delay", 0.0))
    except ValueError: user_delay = 0.0
    
    mode = request.form.get("mode", "auto")

    if not video.filename: return jsonify(error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"), 400
    if Path(secure_filename(video.filename)).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error="MP4, MOV, MKV သို့မဟုတ် WEBM ဖိုင်ကို ရွေးပါ။"), 400

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not groq_key or not gemini_key: return jsonify(error="GROQ_API_KEY နဲ့ GEMINI_API_KEY ထည့်ပါ။"), 500

    try:
        from groq import Groq
        import google.generativeai as genai
    except ImportError: return jsonify(error="Libraries များ မတပ်ဆင်ရသေးပါ။"), 500

    job_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
    video_path = job_dir / secure_filename(video.filename)
    audio_path = job_dir / "audio.mp3"

    try:
        video.save(video_path)
        
        # Determine extraction mode
        use_ocr = False
        if mode == "ocr": use_ocr = True
        elif mode == "auto": use_ocr = check_has_hardcoded_subs(video_path)
        
        raw_segments = []
        source_type = ""
        
        if use_ocr:
            try:
                raw_segments = run_ocr_extraction(video_path)
                source_type = "OCR ဖြင့် စာတန်းကိုတိုက်ရိုက်ဖတ်ယူထားပါသည်။"
            except Exception as e:
                app.logger.exception("OCR Extraction Failed")
                raw_segments = []
                
        if not raw_segments:
            # Fallback to Audio / Whisper
            source_type = "အသံဖမ်းစနစ်(Whisper) ဖြင့် ဖတ်ယူထားပါသည်။"
            subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)], capture_output=True, text=True, timeout=1800)
            if audio_path.stat().st_size == 0: return jsonify(error="ဗီဒီယိုထဲမှာ အသံမတွေ့ပါ။"), 400
            
            groq_client = Groq(api_key=groq_key)
            with open(audio_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(file=(audio_path.name, audio_file.read()), model="whisper-large-v3", language="zh", response_format="verbose_json", timestamp_granularities=["word", "segment"])
                
            for segment in get_field(transcription, "segments", []) or []:
                text = str(get_field(segment, "text", "") or "").strip()
                if not text: continue
                start = float(get_field(segment, "start", 0))
                end = float(get_field(segment, "end", start))
                
                total_delay = SUBTITLE_DELAY_SECONDS + user_delay
                raw_segments.append({"start": max(0.0, start + total_delay), "end": max(start + 0.15, end + total_delay), "text": text})

            if not raw_segments: return jsonify(error="စကားပြောစာသား မတွေ့ပါ။"), 422
            raw_segments.sort(key=lambda item: (item["start"], item["end"]))
            for index in range(len(raw_segments) - 1):
                if raw_segments[index]["end"] > raw_segments[index + 1]["start"]:
                    raw_segments[index]["end"] = max(raw_segments[index]["start"] + 0.15, raw_segments[index + 1]["start"])

        # Translate with Gemini
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        translated_lines = []
        for start_index in range(0, len(raw_segments), GEMINI_BATCH_SIZE):
            batch = raw_segments[start_index:start_index + GEMINI_BATCH_SIZE]
            try: translated_lines.extend(translate_batch(model, batch))
            except Exception: return jsonify(error="မြန်မာဘာသာပြန်ခြင်း မအောင်မြင်ပါ။"), 502

        items = []
        for index, segment in enumerate(raw_segments):
            items.append({
                "index": index + 1,
                "time": f"{make_timestamp(segment['start'])} --> {make_timestamp(segment['end'])}",
                "text": format_subtitle(translated_lines[index]),
                "source": segment["text"]
            })

        return jsonify(items=items, notice=f"မြန်မာဘာသာပြန်ပြီးပါပြီ။ {source_type} စာတန်းများကို ပြန်စစ်ပြီး SRT ထုတ်ပါ။")

    except Exception as e:
        app.logger.error(f"Global Error: {traceback.format_exc()}")
        return jsonify(error="စနစ်ချို့ယွင်းမှု ဖြစ်ပေါ်သွားပါသည်။ Console Log ကို စစ်ဆေးပါ။"), 500
    finally: shutil.rmtree(job_dir, ignore_errors=True)

@app.post("/api/export")
def export_srt():
    items = (request.get_json(silent=True) or {}).get("items", [])
    output = []
    for item in items:
        timestamp, text = str(item.get("time", "")).strip(), str(item.get("text", "")).strip()
        if "-->" in timestamp and text: output.append(f"{len(output) + 1}\n{timestamp}\n{text}")
    return send_file(BytesIO(("\n\n".join(output) + "\n").encode("utf-8-sig")), mimetype="application/x-subrip", as_attachment=True, download_name="myanmar_subtitles.srt")

@app.errorhandler(500)
def server_error(e):
    return jsonify(error="Internal Server Error: Backend တွင် ပြဿနာရှိနေပါသည်။"), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify(error="ဖိုင်အရွယ်အစား ကြီးလွန်းပါသည်။"), 413

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
