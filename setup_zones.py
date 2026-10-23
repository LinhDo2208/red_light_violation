import argparse
import json
import sys

import cv2

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Chiều rộng cửa sổ vẽ (pixel màn hình). Đặt 0 để OpenCV tự chọn. Cũng có thể kéo viền cửa sổ bằng chuột.
WIN_WIDTH = 1280

line_pts = []
roi_list = []
mode = None


def parse_args():
    ap = argparse.ArgumentParser(description="Vẽ vạch + ROI đèn tín hiệu và lưu ra file JSON")
    ap.add_argument("-s", "--source", default="video - Trim1.mp4",
                    help="file video, ảnh (.jpg/.png/...) hoặc RTSP URL")
    ap.add_argument("-o", "--out", default="zones.json",
                    help="file JSON đầu ra (mặc định: zones.json)")
    return ap.parse_args()


def load_first_frame(src):
    if str(src).lower().endswith(IMG_EXT):
        return cv2.imread(src)
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def on_mouse(event, x, y, flags, param):
    global mode
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if mode == "line":
        line_pts.append((x, y))
        if len(line_pts) == 2:
            mode = None
    elif mode == "roi":
        if not roi_list or len(roi_list[-1]) == 2:
            roi_list.append([])
        roi_list[-1].append((x, y))
        if len(roi_list[-1]) == 2:
            mode = None


def put_text(img, txt, org, color=(0, 0, 0), scale=0.7, thick=2):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def main():
    global mode
    args = parse_args()

    frame = load_first_frame(args.source)
    if frame is None:
        print(f"Không đọc được frame từ: {args.source}")
        sys.exit(1)
    H, W = frame.shape[:2]
    print(f"Nguồn: {W}x{H}")

    # Cỡ chữ / nét vẽ tự co giãn theo kích thước ảnh
    ui = min(2.5, max(0.45, H / 720.0))
    font_scale = 0.7 * ui
    thick = max(1, int(round(2 * ui)))
    dot_r = max(3, int(round(4 * ui)))
    line_gap = int(round(28 * ui))

    # Cửa sổ đổi kích thước được; tọa độ chuột luôn tính theo pixel ảnh nên vẫn chính xác
    cv2.namedWindow("setup", cv2.WINDOW_NORMAL)
    if WIN_WIDTH > 0:
        cv2.resizeWindow("setup", WIN_WIDTH, int(round(WIN_WIDTH * H / W)))
    cv2.setMouseCallback("setup", on_mouse)

    while True:
        d = frame.copy()
        for p in line_pts:
            cv2.circle(d, p, dot_r, (0, 0, 255), -1)
        if len(line_pts) == 2:
            cv2.line(d, line_pts[0], line_pts[1], (0, 0, 255), thick)
        for i, r in enumerate(roi_list):
            if len(r) == 1:
                cv2.circle(d, r[0], dot_r, (150, 255, 150), -1)
            elif len(r) == 2:
                cv2.rectangle(d, r[0], r[1], (150, 255, 150), thick)
                put_text(d, f"ROI #{i + 1}",
                         (min(r[0][0], r[1][0]) + 6, min(r[0][1], r[1][1]) + int(20 * ui)),
                         (0, 255, 0), 0.6 * ui, thick)
        y = line_gap
        put_text(d, "Press L then click 2 points to draw LINE", (18, y), (0, 0, 0), font_scale, thick)
        y += line_gap
        put_text(d, "Press R then click 2 points to add TL ROI (multi allowed)", (18, y), (0, 0, 0), font_scale, thick)
        y += line_gap
        put_text(d, f"U: undo last ROI/LINE   C: clear ROIs   SPACE: save   Q/Esc: quit     [mode={mode}]",
                 (18, y), (0, 0, 0), font_scale, thick)
        cv2.imshow("setup", d)

        k = cv2.waitKey(20) & 0xFF
        if k == ord("l"):
            line_pts.clear()
            mode = "line"
        elif k == ord("r"):
            if roi_list and len(roi_list[-1]) == 1:
                roi_list.pop()
            mode = "roi"
        elif k == ord("u"):
            if roi_list:
                roi_list.pop()
            elif line_pts:
                line_pts.clear()
        elif k == ord("c"):
            roi_list.clear()
        elif k == ord(" "):
            rois = [r for r in roi_list if len(r) == 2]
            if len(line_pts) == 2 and rois:
                data = {
                    "frame_size": [W, H],
                    "line": [list(p) for p in line_pts],
                    "rois": [[list(p) for p in r] for r in rois],
                }
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"Đã lưu {args.out}  (frame {W}x{H}, {len(rois)} ROI)")
                break
            else:
                print("Cần vẽ đủ 1 vạch (2 điểm) và ít nhất 1 ROI (2 góc) trước khi lưu.")
        elif k in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()