import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
LETTER_FOLDER = os.path.join(STATIC_FOLDER, "letters")
MODEL_PATH = os.path.join(BASE_DIR, "model1.h5")

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(LETTER_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='templates')

# --- TẢI MODEL ---
try:
    model = load_model(MODEL_PATH)
    print(f"* Model '{MODEL_PATH}' đã tải thành công.")
except Exception as e:
    print(f"* Lỗi khi tải model: {e}")
    model = None

# --- NHÃN ---
LABELS = ['0','1','2','3','4','5','6','7','8','9',
          'A','B','C','D','E','F','G','H','circle',
          'I','J','K','L','M','N','O','P','Q','R',
          'S','square','T','triangle','U','V','W','X','Y','Z']

# --- HÀM LẤY TOP N DỰ ĐOÁN ---
def get_top_predictions(predictions, top_n=5):
    """
    Lấy top N dự đoán có xác suất cao nhất
    predictions: array xác suất từ model (shape: (num_classes,))
    top_n: số lượng dự đoán muốn lấy
    return: list các dict {"label": str, "confidence": float}
    """
    # Lấy indices của top_n giá trị cao nhất
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    top_results = []
    for idx in top_indices:
        top_results.append({
            "label": LABELS[idx],
            "confidence": float(predictions[idx])
        })
    
    return top_results

# --- HÀM TIỀN XỬ LÝ ẢNH MỖI KÝ TỰ ---
def preprocessing_for_prediction(crop_binary, out_size=(64, 64), pad=10):
    """
    Tiền xử lý ảnh một ký tự ĐÃ BINARY để model dự đoán.
    crop_binary: ảnh đã threshold (0 hoặc 255)
    """
    # 1. Đảm bảo ảnh là grayscale
    if crop_binary.ndim == 3:
        crop_binary = cv2.cvtColor(crop_binary, cv2.COLOR_BGR2GRAY)
    
    # 2. Tìm bounding box của nội dung thật sự
    coords = cv2.findNonZero(crop_binary)
    if coords is None:
        # Nếu không có pixel trắng, trả về ảnh trống
        final_img = np.zeros(out_size, dtype=np.float32)
        return np.expand_dims(final_img, axis=-1)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # 3. Crop chặt vùng có nội dung
    tight_crop = crop_binary[y:y+h, x:x+w]
    
    # 4. Thêm padding
    pad_h = int(h * 0.25)  # 25% chiều cao
    pad_w = int(w * 0.25)  # 25% chiều rộng
    padded = cv2.copyMakeBorder(
        tight_crop, 
        pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, 
        value=0
    )
    
    # 5. Resize giữ tỉ lệ
    h_pad, w_pad = padded.shape
    scale = min(out_size[0] / h_pad, out_size[1] / w_pad)
    new_w = int(w_pad * scale)
    new_h = int(h_pad * scale)
    resized = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 6. Đặt vào giữa canvas đen
    canvas = np.zeros(out_size, dtype=np.uint8)
    x_offset = (out_size[1] - new_w) // 2
    y_offset = (out_size[0] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 7. Chuẩn hóa 0-1
    final_img = canvas.astype("float32") / 255.0
    return np.expand_dims(final_img, axis=-1)  # (64,64,1)

# --- HÀM SẮP XẾP CONTOURS THEO DÒNG ---
def sort_contours_by_line(contours):
    """
    Sắp xếp contours theo dòng (top-to-bottom, left-to-right)
    """
    if not contours:
        return []
    
    # Tính chiều cao trung bình
    avg_height = sum([h for (x,y,w,h,cnt) in contours]) / len(contours)
    
    # Nhóm các contours thành dòng
    lines = []
    contours_sorted_by_y = sorted(contours, key=lambda c: c[1])
    
    current_line = [contours_sorted_by_y[0]]
    current_y = contours_sorted_by_y[0][1]
    
    for cont in contours_sorted_by_y[1:]:
        x, y, w, h, cnt = cont
        if abs(y - current_y) < avg_height / 2:
            current_line.append(cont)
        else:
            current_line.sort(key=lambda c: c[0])
            lines.append(current_line)
            current_line = [cont]
            current_y = y
    
    current_line.sort(key=lambda c: c[0])
    lines.append(current_line)
    
    result = []
    for line in lines:
        result.extend(line)
    
    return result

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if model is None:
        return jsonify({"error": "Model chưa được tải!"}), 500
    
    steps = []
    letters = []
    predictions = []

    # ---- Nhận ảnh ----
    if "image" in request.files:
        file = request.files["image"]
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        data = request.get_json()
        if data is None or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        base64_str = data["image"].split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    print(f" Sử dụng Otsu threshold")

    # PIL → OpenCV
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ---- Step 1: Gray ----
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{STATIC_FOLDER}/step_gray.png", gray)
    steps.append({"title": "Grayscale", "file": "step_gray.png"})

    # ---- Step 2: Blur ----
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imwrite(f"{STATIC_FOLDER}/step_blur.png", blur)
    steps.append({"title": "Gaussian Blur", "file": "step_blur.png"})

    # ---- Step 3: Threshold (Otsu) ----
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{STATIC_FOLDER}/step_thresh_otsu.png", th_otsu)
    steps.append({"title": "Threshold (Otsu)", "file": "step_thresh_otsu.png"})
    
    # ---- Step 3.5: Morphology để làm sạch ----
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th_morphed = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    th_morphed = cv2.morphologyEx(th_morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(f"{STATIC_FOLDER}/step_morphology.png", th_morphed)
    steps.append({"title": "Morphology (Clean)", "file": "step_morphology.png"})
    
    # Sử dụng ảnh đã làm sạch để phát hiện và crop
    th_detect = th_morphed

    # ---- Step 4: Tách ký tự ----
    contours, _ = cv2.findContours(th_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    img_area = th_detect.shape[0] * th_detect.shape[1]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        filtered_contours.append((x, y, w, h, cnt))
    
    filtered_contours = sort_contours_by_line(filtered_contours)
    
    img_with_boxes = cv_img.copy()
    for idx, (x, y, w, h, cnt) in enumerate(filtered_contours):
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img_with_boxes, f"#{idx}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imwrite(f"{STATIC_FOLDER}/step_boxes.png", img_with_boxes)
    steps.append({"title": "Detected Characters", "file": "step_boxes.png"})

    # ---- Xóa files cũ ----
    if os.path.exists(LETTER_FOLDER):
        for filename in os.listdir(LETTER_FOLDER):
            file_path = os.path.join(LETTER_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f" Không thể xóa {file_path}: {e}")
    else:
        os.makedirs(LETTER_FOLDER, exist_ok=True)

    # ---- Step 5: Xử lý từng ký tự ----
    for idx, (x, y, w, h, cnt) in enumerate(filtered_contours):
        crop = th_detect[y:y+h, x:x+w]
        processed = preprocessing_for_prediction(crop)

        save_path = os.path.join(LETTER_FOLDER, f"letter_{idx}.png")
        cv2.imwrite(save_path, (processed.squeeze() * 255))

        letters.append(f"letter_{idx}.png")

        # Dự đoán và lấy top 5
        inp = np.expand_dims(processed, axis=0)
        pred = model.predict(inp, verbose=0)[0]  # Lấy array xác suất
        
        # Lấy top 5 dự đoán
        top_preds = get_top_predictions(pred, top_n=5)
        
        predictions.append({
            "top_prediction": top_preds[0]["label"],  # Dự đoán cao nhất
            "confidence": top_preds[0]["confidence"],
            "top_5": top_preds  # Danh sách top 5
        })

    print(f" Tổng số ký tự: {len(letters)}")
    print(f" Top predictions: {[p['top_prediction'] for p in predictions]}")
    
    return jsonify({
        "steps": steps,
        "letters": letters,
        "predictions": predictions
    })

# --- RUN APP ---
if __name__ == "__main__":
    app.run(debug=True)
