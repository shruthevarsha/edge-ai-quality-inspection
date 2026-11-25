#!/usr/bin/env python3
"""NethraQC — offline PCB inspection assistant. Local web UI on http://<device>:8080

  python app.py                       # uses config.yaml
  python app.py --source 0            # USB camera
  python app.py --source picamera2    # Raspberry Pi camera module
  python app.py --source path/to/images --port 8080
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from nethra.core import load_config
from nethra.pipeline import InspectionPipeline

ROOT = Path(__file__).resolve().parent


def create_app(cfg: dict) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    pipe = InspectionPipeline(cfg)
    app.config["pipeline"] = pipe
    data_dir = Path(cfg["app"]["data_dir"]).resolve()

    def _decode(file_storage) -> np.ndarray | None:
        if not file_storage:
            return None
        buf = np.frombuffer(file_storage.read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None

    def _media_url(p: str | None) -> str | None:
        if not p:
            return None
        try:
            return "/media/" + str(Path(p).resolve().relative_to(data_dir))
        except ValueError:
            return None

    def _public(rec: dict | None) -> dict | None:
        if not rec:
            return None
        r = dict(rec)
        r["image_url"] = _media_url(r.get("image_path"))
        r["annotated_url"] = _media_url(r.get("annotated_path"))
        r["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
        return r

    @app.get("/")
    def index():
        return render_template("index.html", station=cfg["app"]["station_id"], sound=cfg["alerts"].get("sound", True))

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            last = -1
            while True:
                jpg, seq = pipe.get_view_jpeg()
                if jpg is None or seq == last:
                    time.sleep(0.03)
                    continue
                last = seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        jpg, _ = pipe.get_view_jpeg(90)
        return Response(jpg or b"", mimetype="image/jpeg")

    @app.get("/api/status")
    def status():
        return jsonify(pipe.status())

    @app.get("/api/latest")
    def latest():
        return jsonify(_public(pipe.latest))

    @app.post("/api/inspect")
    def inspect_now():
        pipe.request_inspect()
        return jsonify({"ok": True})

    @app.post("/api/next")
    def next_part():
        pipe.next_part()
        return jsonify({"ok": True})

    @app.post("/api/pause")
    def pause():
        pipe.paused = bool((request.get_json(silent=True) or {}).get("paused", not pipe.paused))
        return jsonify({"paused": pipe.paused})

    @app.post("/api/upload")
    def upload():
        img = _decode(request.files.get("image"))
        if img is None:
            return jsonify({"error": "no image"}), 400
        ref = _decode(request.files.get("reference"))
        rec = pipe.inspect(img, ref if ref is not None else pipe.reference, f"upload:{request.files['image'].filename}")
        return jsonify(_public(rec))

    @app.post("/api/reference")
    def set_reference():
        img = _decode(request.files.get("image"))
        if img is not None:
            pipe.set_reference(img)
            ok = True
        else:
            ok = pipe.set_reference_from_frame()
        return jsonify({"ok": bool(ok), "reference_set": pipe.reference is not None})

    @app.delete("/api/reference")
    def clear_reference():
        pipe.set_reference(None)
        return jsonify({"ok": True})

    @app.get("/api/inspections")
    def inspections():
        verdict = request.args.get("verdict") or None
        limit = min(int(request.args.get("limit", 50)), 500)
        return jsonify([_public(r) for r in pipe.store.list(limit=limit, verdict=verdict)])

    @app.get("/api/inspections/<int:iid>")
    def inspection(iid: int):
        r = pipe.store.get(iid)
        return jsonify(_public(r)) if r else (jsonify({"error": "not found"}), 404)

    @app.post("/api/inspections/<int:iid>/action")
    def operator_action(iid: int):
        body = request.get_json(force=True)
        action = body.get("action")
        if action not in ("confirm", "override_pass", "send_rework", "reject"):
            return jsonify({"error": "bad action"}), 400
        rec = pipe.store.set_operator(iid, action, body.get("note", ""))
        if pipe.latest and pipe.latest["id"] == iid:
            pipe.latest = rec
        return jsonify(_public(rec))

    @app.get("/api/stats")
    def stats():
        hours = float(request.args.get("hours", 8))
        return jsonify(pipe.store.stats(time.time() - hours * 3600, station=pipe.station))

    @app.post("/api/ask")
    def ask():
        body = request.get_json(force=True)
        q = (body.get("question") or "").strip()
        if not q:
            return jsonify({"error": "empty question"}), 400
        ctx = {"stats": pipe.store.stats(station=pipe.station)}
        iid = body.get("inspection_id") or (pipe.latest or {}).get("id")
        if iid:
            ctx["inspection"] = pipe.store.get(int(iid))
        return jsonify({"job": pipe.advisor.submit_question(q, ctx)})

    @app.post("/api/report")
    def report():
        hours = float((request.get_json(silent=True) or {}).get("hours", 8))
        st = pipe.store.stats(time.time() - hours * 3600, station=pipe.station)

        def done(res: dict) -> None:
            if "summary" in res:
                pipe.store.save_report(st, res["summary"], res.get("backend", ""))

        return jsonify({"job": pipe.advisor.submit_report(st, done), "stats": st})

    @app.get("/api/jobs/<jid>")
    def job(jid: str):
        j = pipe.advisor.job(jid)
        return jsonify(j) if j else (jsonify({"error": "unknown job"}), 404)

    @app.get("/api/export.csv")
    def export_csv():
        hours = request.args.get("hours")
        since = time.time() - float(hours) * 3600 if hours else None
        return Response(pipe.store.export_csv(since), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=nethraqc_{time.strftime('%Y%m%d_%H%M')}.csv"})

    @app.get("/media/<path:rel>")
    def media(rel: str):
        p = (data_dir / rel).resolve()
        if data_dir not in p.parents or not p.is_file():
            abort(404)
        return send_file(p)

    return app


def ensure_demo_samples(cfg: dict) -> None:
    """The demo board images are generated on first run instead of being stored in the repository."""
    src = Path(str(cfg["camera"]["source"]))
    demo = ROOT / "samples" / "demo"
    if src.resolve() == demo.resolve() and not any(demo.glob("*_test.*")):
        logging.info("Generating synthetic demo boards in %s (first run only)", demo)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_samples.py"), "--out", str(demo)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="NethraQC offline inspection assistant")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="camera index, 'picamera2', video file or image folder")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="use the template advisor (UI development only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ov: dict = {}
    if a.source is not None:
        ov.setdefault("camera", {})["source"] = a.source
    if a.no_llm:
        ov.setdefault("advisor", {})["backend"] = "template"
    cfg = load_config(a.config, ov)
    ensure_demo_samples(cfg)
    app = create_app(cfg)
    host, port = a.host or cfg["app"]["host"], a.port or cfg["app"]["port"]
    logging.info("NethraQC UI: http://%s:%d", "localhost" if host == "0.0.0.0" else host, port)
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
