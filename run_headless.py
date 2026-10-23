"""
run_headless.py  -  CHẠY TRÊN NODE EDGE (không cần màn hình).

Đọc zones.json (tạo bằng setup_zones.py), phát hiện xe vượt vạch khi đèn đỏ.
Khi có vi phạm: lưu ảnh + ghi 1 dòng vào events.jsonl (+ gửi webhook nếu có).

Ví dụ:
    python run_headless.py --source "video - Trim.mp4" --save-video output_violations.mp4
    python run_headless.py --source rtsp://user:pass@192.168.1.10:554/stream
    python run_headless.py --source rtsp://... --weights yolov8n.engine --light-weights best.pt --imgsz 640 --vid-stride 2
    python run_headless.py --source rtsp://... --webhook http://server:8000/violation

Cài đặt:
    pip install ultralytics opencv-python-headless numpy
"""
import argparse
import json
import os
import signal
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------- Model nhan dien mau den -----------------------------
LIGHT_MODEL = None   # duoc nap trong run()
LIGHT_KW = {}        # tham so predict cho model den (imgsz, conf, device)

VEH_OK = {"car", "bus", "truck", "motorcycle"}

COLOR_LINE = (0, 0, 255)
COLOR_TEXT = (230, 230, 230)
COLOR_BAD = (0, 0, 255)
COLOR_ROI = (150, 255, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX

STOP = False


def _handle_stop(signum, frame):
    global STOP
    STOP = True
    print("\nNhận tín hiệu dừng, đang thoát...")


signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)


# ----------------------------- Tiện ích vẽ -----------------------------
def put_text(img, txt, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_stats_panel(frame, violations_count):
    panel_w, panel_h = 260, 70
    x0, y0 = 15, 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (25, 25, 25), -1)
    frame[y0:y0 + panel_h, x0:x0 + panel_w] = cv2.addWeighted(
        overlay[y0:y0 + panel_h, x0:x0 + panel_w], 0.45,
        frame[y0:y0 + panel_h, x0:x0 + panel_w], 0.55, 0
    )
    put_text(frame, "Violations:", (x0 + 12, y0 + 28), COLOR_TEXT, 0.7, 2)
    put_text(frame, f"{violations_count}", (x0 + 12, y0 + 58), COLOR_BAD, 0.95, 2)


# ----------------------------- Nhận diện màu đèn -----------------------------
def model_major_color(res):
    """Lay mau co conf cao nhat trong ket qua detect cua 1 ROI."""
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "unknown"
    k = int(boxes.conf.argmax())
    name = str(res.names[int(boxes.cls[k])]).lower()
    return name if name in ("red", "yellow", "green") else "unknown"


def crop_roi(frame, roi_pts):
    (x1, y1), (x2, y2) = roi_pts
    xa, ya = max(0, min(x1, x2)), max(0, min(y1, y2))
    xb, yb = min(frame.shape[1] - 1, max(x1, x2)), min(frame.shape[0] - 1, max(y1, y2))
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()


def infer_all_roi_colors(frame, rois):
    colors = ["unknown"] * len(rois)
    crops = [crop_roi(frame, roi) for roi in rois]
    valid = [i for i, c in enumerate(crops) if c is not None and c.size > 0]
    if not valid:
        return colors
    results = LIGHT_MODEL.predict([crops[i] for i in valid], verbose=False, **LIGHT_KW)  # 1 lan goi cho tat ca ROI
    for i, r in zip(valid, results):
        colors[i] = model_major_color(r)
    return colors


# ----------------------------- Hình học -----------------------------
def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


# ----------------------------- Cấu hình vùng -----------------------------
def load_zones(path, W, H):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sw, sh = cfg["frame_size"]
    sx, sy = W / sw, H / sh  # tự scale nếu độ phân giải khác lúc setup
    line = [(int(x * sx), int(y * sy)) for x, y in cfg["line"]]
    rois = [[(int(x * sx), int(y * sy)) for x, y in r] for r in cfg["rois"]]
    return line, rois


def is_live_source(src):
    s = str(src).lower()
    return s.startswith(("rtsp://", "rtmp://", "http://", "https://")) or s.isdigit()


def probe_source(src, retries=10, wait=3.0):
    """Mở nguồn, đọc frame đầu để lấy kích thước + fps. Thử lại nếu là stream."""
    src_cv = int(src) if str(src).isdigit() else src
    for i in range(retries):
        cap = cv2.VideoCapture(src_cv)
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()
        if ok:
            H, W = frame.shape[:2]
            return W, H, fps
        print(f"[probe] Không đọc được nguồn (lần {i + 1}/{retries}), thử lại sau {wait}s...")
        if STOP:
            break
        time.sleep(wait)
    raise RuntimeError(f"Không mở được nguồn video: {src}")


# ----------------------------- Gửi sự kiện -----------------------------
def post_webhook(url, payload):
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:
        print(f"[webhook] lỗi: {e}")


def report_violation(args, frame, bbox, tid, cls_name, frame_idx):
    ts = datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.events_dir, exist_ok=True)
    img_path = os.path.join(args.events_dir, f"violation_{stamp}_id{tid}.jpg")
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    event = {
        "time": ts.isoformat(timespec="seconds"),
        "track_id": int(tid),
        "class": cls_name,
        "bbox": [int(v) for v in bbox],
        "frame": int(frame_idx),
        "image": img_path,
        "source": args.source,
    }
    with open(os.path.join(args.events_dir, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[VIOLATION] {event}")

    if args.webhook:
        threading.Thread(target=post_webhook, args=(args.webhook, event), daemon=True).start()


# ----------------------------- Vòng chạy chính -----------------------------
def run(args):
    live = is_live_source(args.source)
    if not live and not os.path.exists(args.source):
        raise FileNotFoundError(f"Input video not found: {args.source}")

    W, H, fps = probe_source(args.source)
    line_pts, rois = load_zones(args.config, W, H)
    p1, p2 = line_pts[0], line_pts[1]
    print(f"Nguồn {W}x{H} @ {fps:.1f}fps | vạch {p1}->{p2} | {len(rois)} ROI")

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = YOLO(args.weights)

    global LIGHT_MODEL, LIGHT_KW
    LIGHT_MODEL = YOLO(args.light_weights)
    LIGHT_KW = dict(imgsz=args.light_imgsz, conf=args.light_conf)
    if args.device:
        LIGHT_KW["device"] = args.device

    violated_ids = set()
    prev_center = {}
    last_seen = {}
    roi_colors = []
    names_cache = None
    frame_idx = -1
    track_kwargs = dict(
        conf=args.conf, iou=args.iou, tracker=args.tracker,
        stream=True, persist=True, verbose=False, imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )
    if args.device:
        track_kwargs["device"] = args.device

    try:
        while not STOP:
            try:
                src = int(args.source) if str(args.source).isdigit() else args.source
                stream = model.track(source=src, **track_kwargs)
                for res in stream:
                    if STOP:
                        break
                    frame_idx += 1
                    frame = res.orig_img.copy()

                    if names_cache is None and isinstance(getattr(res, "names", None), dict):
                        names_cache = {int(k): v for k, v in res.names.items()}

                    if frame_idx % args.light_refresh == 0 or not roi_colors:
                        roi_colors = infer_all_roi_colors(frame, rois)
                    any_red = any(c == "red" for c in roi_colors)

                    # Vẽ chú thích
                    cv2.line(frame, p1, p2, COLOR_LINE, 2)
                    for idx, roi in enumerate(rois):
                        (x1, y1), (x2, y2) = roi
                        rx1, ry1, rx2, ry2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_ROI, 2)
                        c = roi_colors[idx] if idx < len(roi_colors) else "unknown"
                        put_text(frame, c.upper(), (rx1, max(20, ry1 - 8)), COLOR_ROI, 0.7, 2)

                    new_violations = []
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
                        clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
                        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

                        for bb, tid, cidx, cf in zip(xyxy, ids, clses, confs):
                            tid = int(tid)
                            cls_name = names_cache.get(int(cidx), str(cidx)) if names_cache else str(cidx)
                            if cls_name not in VEH_OK or cf < args.conf:
                                continue

                            x1, y1, x2, y2 = map(int, bb)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            last_seen[tid] = frame_idx

                            if tid not in prev_center:
                                prev_center[tid] = (cx, cy)
                            crossed_now = segments_intersect(prev_center[tid], (cx, cy), p1, p2)
                            prev_center[tid] = (cx, cy)

                            if any_red and crossed_now and tid not in violated_ids:
                                violated_ids.add(tid)
                                new_violations.append((tid, cls_name, (x1, y1, x2, y2)))

                            if tid in violated_ids:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BAD, 3)
                                put_text(frame, "VIOLATION", (x1, max(22, y1 - 10)), COLOR_BAD, 0.8, 2)
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
                                put_text(frame, f"{cls_name} ID{tid}", (x1, max(22, y1 - 10)), (180, 255, 180), 0.6, 2)

                    draw_stats_panel(frame, len(violated_ids))

                    for tid, cls_name, bbox in new_violations:
                        report_violation(args, frame, bbox, tid, cls_name, frame_idx)

                    if writer is not None:
                        writer.write(frame)

                    # Dọn các track đã biến mất để không phình bộ nhớ khi chạy 24/7
                    if frame_idx % 600 == 0:
                        stale = [t for t, f in last_seen.items() if frame_idx - f > 900]
                        for t in stale:
                            last_seen.pop(t, None)
                            prev_center.pop(t, None)

                    if frame_idx % 300 == 0:
                        print(f"[frame {frame_idx}] đèn={roi_colors} | vi phạm={len(violated_ids)}")

                # stream kết thúc
                if not live:
                    break  # file video: chạy xong thì thoát
                print("[stream] Mất kết nối, thử kết nối lại sau 3s...")
                time.sleep(3)

            except Exception as e:
                if not live:
                    raise
                print(f"[stream] Lỗi: {e}. Thử kết nối lại sau 5s...")
                time.sleep(5)
    finally:
        if writer is not None:
            writer.release()
        print(f"Kết thúc. Tổng số vi phạm: {len(violated_ids)}")


def parse_args():
    ap = argparse.ArgumentParser(description="Phát hiện vượt đèn đỏ - chạy headless trên edge")
    ap.add_argument("--source", default="video - Trim.mp4", help="file video, RTSP URL hoặc index webcam")
    ap.add_argument("--config", default="zones.json", help="file vùng tạo bởi setup_zones.py")
    ap.add_argument("--weights", default="yolov8n.pt", help="yolov8n.pt / .onnx / .engine ...")
    ap.add_argument("--light-weights", default="best.pt", help="model detect mau den (green/red/yellow)")
    ap.add_argument("--light-imgsz", type=int, default=640)
    ap.add_argument("--light-conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--vid-stride", type=int, default=1, help="bỏ qua frame để giảm tải (2 = xử lý 1/2 số frame)")
    ap.add_argument("--device", default=None, help="vd: cpu, 0 (GPU/Jetson)")
    ap.add_argument("--light-refresh", type=int, default=5, help="cập nhật màu đèn mỗi N frame")
    ap.add_argument("--save-video", default=None, help="ghi video đã chú thích (bỏ trống = không ghi)")
    ap.add_argument("--events-dir", default="events", help="thư mục lưu ảnh vi phạm + events.jsonl")
    ap.add_argument("--webhook", default=None, help="URL nhận POST JSON khi có vi phạm")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
