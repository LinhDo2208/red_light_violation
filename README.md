# Phát hiện vượt đèn đỏ (Edge)

Quy trình gồm 2 bước, chạy trên 2 máy khác nhau:

| Bước | Máy | File chạy | Kết quả |
|------|-----|-----------|---------|
| 1. Vẽ vạch dừng + khung đèn | Máy có màn hình (PC/laptop) | `setup_zones.py` | `zones.json` |
| 2. Phát hiện vi phạm | Node edge (không màn hình) | `run_headless.py` | ảnh vi phạm + `events.jsonl` |

---

## Cấu trúc file

```
Máy có màn hình:            Node edge:
├── setup_zones.py          ├── run_headless.py
└── video - Trim.mp4        ├── zones.json        <- copy từ máy bên trái
                            ├── best.pt           <- model nhận diện màu đèn
                            ├── yolov8n.pt        <- model phát hiện xe
                            └── video - Trim.mp4  <- CÙNG video đã dùng ở bước 1
```

> **Quan trọng:** video (hoặc stream) ở bước 1 và bước 2 phải là **cùng một nguồn / cùng góc camera**. Nếu khác, vạch và khung đèn sẽ lệch vị trí.

---

## Bước 1 — Vẽ vạch và khung đèn (máy có màn hình)

### Cài đặt
```bash
pip install opencv-python numpy
```
(Dùng `opencv-python`, **không** phải `-headless`, vì cần hiện cửa sổ.)

### Chạy
```bash
python setup_zones.py --source "video - Trim.mp4" --out zones.json
```
Nguồn có thể là file video, ảnh (`.jpg/.png`) hoặc RTSP URL. Chương trình lấy **frame đầu tiên** để vẽ.

> Lưu ý: giá trị mặc định của `--source` trong `setup_zones.py` là `video - Trim1.mp4`, còn trong `run_headless.py` là `video - Trim.mp4`. Luôn truyền `--source` rõ ràng để tránh nhầm.

### Thao tác trong cửa sổ `setup`

| Phím | Chức năng |
|------|-----------|
| `L` | Vẽ **vạch dừng**: bấm `L`, rồi click 2 điểm |
| `R` | Thêm **khung đèn (ROI)**: bấm `R`, rồi click 2 góc (đối diện nhau). Có thể thêm nhiều ROI |
| `U` | Hoàn tác ROI cuối (hoặc xóa vạch nếu chưa có ROI) |
| `C` | Xóa toàn bộ ROI |
| `SPACE` | **Lưu** ra `zones.json` và thoát |
| `Q` / `Esc` | Thoát, không lưu |

### Mẹo vẽ
- **Vạch:** đặt ngang qua làn đường, tại vị trí xe phải dừng. Xe bị tính vi phạm khi tâm xe cắt qua vạch lúc đèn đỏ.
- **Khung đèn:** ôm sát cụm đèn, chừa một chút nền xung quanh. Khung quá chật thì model dễ nhận sai.
- Cần ít nhất **1 vạch + 1 ROI** thì mới lưu được.

### Kết quả
File `zones.json` có dạng:
```json
{
  "frame_size": [1920, 1080],
  "line": [[x1, y1], [x2, y2]],
  "rois": [[[x1, y1], [x2, y2]]]
}
```
`run_headless.py` tự scale tọa độ nếu độ phân giải nguồn khác `frame_size`, nhưng nên dùng cùng độ phân giải và cùng tỉ lệ khung hình.

---

## Bước 2 — Chạy trên node edge

### Cài đặt
```bash
pip install ultralytics opencv-python-headless numpy
```

### Copy file .json sang edge
(Cùng với `run_headless.py` và `yolov8n.pt` nếu chưa có.)

### Chạy với file video
```bash
python run_headless.py \
    --source "video - Trim.mp4" \
    --config zones.json \
    --weights yolov8n.pt \
    --light-weights best.pt \
    --save-video output_violations.mp4
```

## Tham số chính

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--source` | `video - Trim.mp4` | File video, RTSP URL hoặc index webcam |
| `--config` | `zones.json` | File vùng từ bước 1 |
| `--weights` | `yolov8n.pt` | Model phát hiện xe (`.pt` / `.onnx` / `.engine`) |
| `--light-weights` | `best.pt` | Model nhận diện màu đèn (green / red / yellow) |
| `--light-imgsz` | `640` | Kích thước ảnh đưa vào model đèn |
| `--light-conf` | `0.25` | Ngưỡng conf của model đèn |
| `--light-refresh` | `5` | Cập nhật màu đèn mỗi N frame |
| `--conf` / `--iou` | `0.35` / `0.45` | Ngưỡng của model xe |
| `--imgsz` | `640` | Kích thước ảnh model xe |
| `--vid-stride` | `1` | Bỏ qua frame để giảm tải (2 = xử lý nửa số frame) |
| `--device` | tự chọn | `cpu`, `0`, ... |
| `--save-video` | tắt | Ghi video đã chú thích |
| `--events-dir` | `events` | Thư mục lưu ảnh vi phạm + `events.jsonl` |
| `--webhook` | tắt | URL nhận POST JSON khi có vi phạm |

---

## Kết quả đầu ra

Trong thư mục `events/`:
- `violation_<thời gian>_id<track>.jpg` — ảnh lúc xe vi phạm
- `events.jsonl` — mỗi vi phạm một dòng JSON:

```json
{"time": "2026-10-23T10:15:30", "track_id": 7, "class": "car",
 "bbox": [100, 200, 300, 400], "frame": 1520,
 "image": "events/violation_20261023_101530_id7.jpg", "source": "video - Trim.mp4"}
```

Dừng chương trình bằng `Ctrl+C` hoặc `SIGTERM`. Với nguồn RTSP, chương trình tự kết nối lại khi mất stream.

---

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
|------------|------------|
| Vạch / khung đèn lệch vị trí | Kiểm tra bước 1 và bước 2 dùng cùng video, cùng góc camera, cùng độ phân giải |
| Màu đèn luôn `UNKNOWN` | Nới rộng ROI cho thoáng hơn, hoặc giảm `--light-conf` (vd `0.15`) |
| Màu đèn bị nhận sai | Vẽ lại ROI ôm sát cụm đèn hơn, hoặc tăng `--light-conf` |
| Không có vi phạm dù xe vượt vạch | Kiểm tra nhãn màu đèn trên video (`--save-video`); vạch phải nằm trên đường đi của tâm xe |