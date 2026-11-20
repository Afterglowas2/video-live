"""Сайт «Оживи фото»: фото -> говорящее видео с твоим звуком.

Провайдер видео выбирается переменной VIDEO_PROVIDER:
  heygen (по умолчанию) -- фото человека + твоя звуковая дорожка -> видео, где человек говорит этим звуком (Avatar IV)
  runway                -- фото + промпт -> нейросеть анимирует кадр, затем накладывается твой звук

Режим audio_only (твоё видео + твой звук) работает без нейросети и без ключей.
"""
import io
import os
import shutil
import subprocess
import threading
import time
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
PROVIDER = os.environ.get("VIDEO_PROVIDER", "heygen").lower()
RATIOS = {"1280:720": "16:9", "720:1280": "9:16", "960:960": "1:1"}
HEYGEN = os.environ.get("sk_V2_hgu_kwTaxqDUO9M_12InUbvb4EnLbI6hHDOnimOFdbvG6bZp", "https://api.heygen.com")
RUNWAY_MODEL = os.environ.get("RUNWAY_MODEL", "gen4_turbo")


def api_key_name() -> str:
    return "HEYGEN_API_KEY" if PROVIDER == "heygen" else "RUNWAYML_API_SECRET"


# ---------- общие помощники ----------
def prepare_image(path: Path) -> bytes:
    """Уменьшаем фото и приводим к JPEG (лимит HeyGen на картинку 32 МБ)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((2048, 2048))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def prepare_audio(src: Path, dst: Path, offset: float, volume: float) -> None:
    """Обрезаем начало, меняем громкость, кодируем в mp3 (HeyGen принимает mp3/wav до 50 МБ)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(max(offset, 0)), "-i", str(src),
         "-vn", "-af", f"volume={volume}", "-c:a", "libmp3lame", "-b:a", "192k", str(dst)],
        check=True, capture_output=True,
    )


def check(r: requests.Response, what: str) -> dict:
    if not r.ok:
        raise RuntimeError(f"{what}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def download(url: str, out: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            shutil.copyfileobj(r.raw, f)


# ---------- HeyGen ----------
def heygen_upload(name: str, data: bytes, mime: str) -> str:
    r = requests.post(f"{HEYGEN}/v3/assets", headers={"X-Api-Key": os.environ["HEYGEN_API_KEY"]},
                      files={"file": (name, data, mime)}, timeout=180)
    return check(r, "Загрузка файла в HeyGen")["data"]["asset_id"]


def generate_heygen(image: Path, audio: Path, prompt: str, ratio: str, out: Path, job: dict) -> None:
    headers = {"x-api-key": os.environ["HEYGEN_API_KEY"], "Content-Type": "application/json"}
    job["message"] = "Загружаю фото и звук в HeyGen…"
    image_id = heygen_upload("photo.jpg", prepare_image(image), "image/jpeg")
    audio_id = heygen_upload("audio.mp3", audio.read_bytes(), "audio/mpeg")

    body = {
        "type": "image",
        "image": {"type": "asset_id", "asset_id": image_id},
        "audio_asset_id": audio_id,
        "aspect_ratio": RATIOS.get(ratio, "16:9"),
        "resolution": os.environ.get("HEYGEN_RESOLUTION", "720p"),
        "expressiveness": os.environ.get("HEYGEN_EXPRESSIVENESS", "medium"),
        "title": "videoalive",
    }
    if prompt:
        body["motion_prompt"] = prompt
    r = requests.post(f"{HEYGEN}/v3/videos", headers=headers, json=body, timeout=60)
    video_id = check(r, "Создание видео в HeyGen")["data"]["video_id"]

    job["message"] = "HeyGen оживляет фото (обычно 2–5 минут)…"
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        time.sleep(5)
        j = check(requests.get(f"{HEYGEN}/v3/videos/{video_id}", headers=headers, timeout=60), "Статус видео")
        d = j.get("data", j)
        st = d.get("status")
        if st == "completed":
            download(d["video_url"], out)
            return
        if st == "failed":
            raise RuntimeError(f"HeyGen: {d.get('failure_message') or d.get('failure_code') or 'ошибка генерации'}")
    raise RuntimeError("HeyGen: превышено время ожидания (20 минут)")


# ---------- Runway ----------
def generate_runway(image: Path, prompt: str, ratio: str, duration: int, out: Path, job: dict) -> None:
    import base64

    from runwayml import RunwayML  # читает ключ из RUNWAYML_API_SECRET

    job["message"] = "Нейросеть оживляет фото (обычно 1–3 минуты)…"
    uri = "data:image/jpeg;base64," + base64.b64encode(prepare_image(image)).decode()
    task = RunwayML().image_to_video.create(
        model=RUNWAY_MODEL, prompt_image=uri,
        prompt_text=prompt or "Natural subtle motion, cinematic",
        ratio=ratio, duration=duration,
    ).wait_for_task_output()
    download(task.output[0], out)


# ---------- звук поверх готового видео ----------
def mux_audio(video: Path, audio: Path, out: Path, offset: float, volume: float) -> None:
    """Заменяет звук видео на загруженную дорожку. Длина = длина видео (короткий звук дополняется тишиной)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-ss", str(max(offset, 0)), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-af", f"volume={volume},apad", "-shortest", "-movflags", "+faststart", str(out)],
        check=True, capture_output=True,
    )


# ---------- задачи ----------
def run_job(job_id: str, form: dict, files: dict[str, Path]) -> None:
    job = JOBS[job_id]
    d = JOBS_DIR / job_id
    result = d / "result.mp4"
    try:
        if form["mode"] == "audio_only":
            job["status"], job["message"] = "audio", "Накладываю звук…"
            mux_audio(files["video"], files["audio"], result, form["offset"], form["volume"])
        elif PROVIDER == "heygen":
            job["status"] = "generating"
            audio = d / "audio_prepared.mp3"
            prepare_audio(files["audio"], audio, form["offset"], form["volume"])
            generate_heygen(files["image"], audio, form["prompt"], form["ratio"], result, job)
        else:
            job["status"] = "generating"
            raw = d / "generated.mp4"
            generate_runway(files["image"], form["prompt"], form["ratio"], form["duration"], raw, job)
            if "audio" in files:
                job["status"], job["message"] = "audio", "Накладываю звук…"
                mux_audio(raw, files["audio"], result, form["offset"], form["volume"])
            else:
                shutil.copy(raw, result)
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


@app.get("/api/config")
def config():
    return jsonify(provider=PROVIDER)


@app.post("/api/jobs")
def create_job():
    mode = request.form.get("mode", "animate")
    if mode not in ("animate", "audio_only"):
        abort(400, "bad mode")
    if mode == "animate" and not os.environ.get(api_key_name()):
        return jsonify(error=f"На сервере не задан {api_key_name()} (ключ API)."), 400

    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir()
    files = {k: p for k in ("image", "video", "audio") if (p := save_upload(k, d))}

    def fail(msg):
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error=msg), 400

    if mode == "animate" and "image" not in files:
        return fail("Не загружено фото.")
    if mode == "audio_only" and "video" not in files:
        return fail("Не загружено видео.")
    if "audio" not in files and (mode == "audio_only" or PROVIDER == "heygen"):
        return fail("Загрузи звуковую дорожку.")

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
