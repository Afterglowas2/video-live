"""Сайт «Оживи фото»: фото -> видео (через Runway API) + твоя звуковая дорожка.

Режимы:
  animate     -- фото + промпт -> нейросеть делает видео, затем (по желанию) накладывается твой звук
  audio_only  -- твоё видео + твой звук -> склейка без нейросети (API-ключ не нужен)
"""
import base64
import io
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image

load_dotenv()

BASE = Path(__file__).parent
JOBS_DIR = BASE / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 МБ на запрос

JOBS: dict[str, dict] = {}
MODEL = os.environ.get("RUNWAY_MODEL", "gen4_turbo")
RATIOS = {"1280:720", "720:1280", "960:960"}


# ---------- нейросеть ----------
def image_to_data_uri(path: Path) -> str:
    """Уменьшаем фото и кодируем в data URI (чтобы не зависеть от публичных ссылок)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((2048, 2048))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_video(image_path: Path, prompt: str, ratio: str, duration: int, out: Path) -> None:
    """Вызов Runway. Чтобы сменить провайдера (Luma, Kling, Replicate…), замени только эту функцию."""
    from runwayml import RunwayML  # читает ключ из RUNWAYML_API_SECRET

    client = RunwayML()
    task = client.image_to_video.create(
        model=MODEL,
        prompt_image=image_to_data_uri(image_path),
        prompt_text=prompt or "Natural subtle motion, cinematic",
        ratio=ratio,
        duration=duration,
    ).wait_for_task_output()
    url = task.output[0]
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            shutil.copyfileobj(r.raw, f)


# ---------- звук ----------
def mux_audio(video: Path, audio: Path, out: Path, offset: float, volume: float) -> None:
    """Заменяет звук видео на загруженную дорожку. Длина = длина видео (короткий звук дополняется тишиной)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-ss", str(max(offset, 0)), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-af", f"volume={volume},apad",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------- задачи ----------
def run_job(job_id: str, form: dict, files: dict[str, Path]) -> None:
    job = JOBS[job_id]
    d = JOBS_DIR / job_id
    try:
        if form["mode"] == "animate":
            job["status"], job["message"] = "generating", "Нейросеть оживляет фото (обычно 1–3 минуты)…"
            raw = d / "generated.mp4"
            generate_video(files["image"], form["prompt"], form["ratio"], form["duration"], raw)
            video = raw
        else:
            video = files["video"]

        result = d / "result.mp4"
        if "audio" in files:
            job["status"], job["message"] = "audio", "Накладываю звук…"
            mux_audio(video, files["audio"], result, form["offset"], form["volume"])
        else:
            shutil.copy(video, result)
        job["status"], job["message"] = "done", "Готово"
        job["url"] = f"/files/{job_id}/result.mp4"
    except subprocess.CalledProcessError as e:
        job["status"], job["message"] = "error", "Ошибка ffmpeg: " + e.stderr.decode(errors="ignore")[-300:]
    except Exception as e:  # noqa: BLE001
        job["status"], job["message"] = "error", f"{type(e).__name__}: {str(e)[:300]}"


def save_upload(field: str, d: Path) -> Path | None:
    f = request.files.get(field)
    if not f or not f.filename:
        return None
    path = d / (field + Path(f.filename).suffix.lower()[:8])
    f.save(path)
    return path


@app.post("/api/jobs")
def create_job():
    mode = request.form.get("mode", "animate")
    if mode not in ("animate", "audio_only"):
        abort(400, "bad mode")
    if mode == "animate" and not os.environ.get("RUNWAYML_API_SECRET"):
        return jsonify(error="На сервере не задан RUNWAYML_API_SECRET (ключ Runway)."), 400

    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir()
    files = {k: p for k in ("image", "video", "audio") if (p := save_upload(k, d))}
    need = "image" if mode == "animate" else "video"
    if need not in files:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error="Не загружен " + ("файл с фото." if need == "image" else "видеофайл.")), 400
    if mode == "audio_only" and "audio" not in files:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error="Загрузи звуковую дорожку."), 400

    ratio = request.form.get("ratio", "1280:720")
    form = {
        "mode": mode,
        "prompt": request.form.get("prompt", "").strip()[:900],
        "ratio": ratio if ratio in RATIOS else "1280:720",
        "duration": 10 if request.form.get("duration") == "10" else 5,
        "offset": float(request.form.get("offset") or 0),
        "volume": min(max(float(request.form.get("volume") or 1), 0), 3),
    }
    JOBS[job_id] = {"status": "queued", "message": "В очереди…"}
    threading.Thread(target=run_job, args=(job_id, form, files), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    return jsonify(job)


@app.get("/files/<job_id>/<name>")
def files(job_id, name):
    if job_id not in JOBS or name != "result.mp4":
        abort(404)
    return send_from_directory(JOBS_DIR / job_id, name, as_attachment=request.args.get("dl") == "1")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
