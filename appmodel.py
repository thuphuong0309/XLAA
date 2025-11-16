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

# --- HÀM TIỀN XỬ LÝ ẢNH MỖI KÝ TỰ (ĐÃ SỬA) ---
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
    pad_h = int(h * 0.2)  # 20% chiều cao
    pad_w = int(w * 0.2)  # 20% chiều rộng
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

# --- HÀM CHUYỂN BASE64 → OPENCV ---
def base64_to_cv2(base64_str):
    base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(img_data)).convert("RGBA")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
    return cv_img

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("indecx.html")

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

    # ---- Step 3: Threshold ----
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{STATIC_FOLDER}/step_thresh.png", th)
    steps.append({"title": "Threshold", "file": "step_thresh.png"})

    # ---- Step 4: Tách ký tự ----
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours quá nhỏ hoặc quá lớn
    filtered_contours = []
    img_area = th.shape[0] * th.shape[1]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Bỏ qua contour quá nhỏ (<1% ảnh) hoặc quá lớn (>80% ảnh)
        if area > img_area * 0.01 and area < img_area * 0.8:
            if w > 10 and h > 10:  # Kích thước tối thiểu
                filtered_contours.append((x, y, w, h, cnt))
    
    # ---- Sắp xếp theo dòng (trên → dưới) rồi trái → phải ----
    def sort_contours_by_line(contours, line_threshold=20):
        """
        Sắp xếp contours theo dòng (top-to-bottom, left-to-right)
        line_threshold: khoảng cách Y để coi là cùng 1 dòng
        """
        if not contours:
            return []
        
        # Tính chiều cao trung bình
        avg_height = sum([h for (x,y,w,h,cnt) in contours]) / len(contours)
        
        # Nhóm các contours thành dòng
        lines = []
        contours_sorted_by_y = sorted(contours, key=lambda c: c[1])  # Sắp theo Y trước
        
        current_line = [contours_sorted_by_y[0]]
        current_y = contours_sorted_by_y[0][1]
        
        for cont in contours_sorted_by_y[1:]:
            x, y, w, h, cnt = cont
            # Nếu Y gần với dòng hiện tại (chênh lệch < avg_height/2)
            if abs(y - current_y) < avg_height / 2:
                current_line.append(cont)
            else:
                # Sắp xếp dòng hiện tại theo X (trái → phải)
                current_line.sort(key=lambda c: c[0])
                lines.append(current_line)
                # Bắt đầu dòng mới
                current_line = [cont]
                current_y = y
        
        # Thêm dòng cuối
        current_line.sort(key=lambda c: c[0])
        lines.append(current_line)
        
        # Gộp tất cả dòng lại
        result = []
        for line in lines:
            result.extend(line)
        
        return result
    
    filtered_contours = sort_contours_by_line(filtered_contours)
    
    # ---- Vẽ bounding boxes lên ảnh gốc ----
    img_with_boxes = cv_img.copy()
    for idx, (x, y, w, h, cnt) in enumerate(filtered_contours):
        # Vẽ khung chữ nhật màu xanh lá
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # Vẽ số thứ tự góc trên bên trái
        cv2.putText(img_with_boxes, f"#{idx}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imwrite(f"{STATIC_FOLDER}/step_boxes.png", img_with_boxes)
    steps.append({"title": "Detected Characters", "file": "step_boxes.png"})

    # ---- Xóa files cũ trong thư mục letters ----
    if os.path.exists(LETTER_FOLDER):
        for filename in os.listdir(LETTER_FOLDER):
            file_path = os.path.join(LETTER_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"🗑️ Đã xóa: {file_path}")
            except Exception as e:
                print(f"⚠️ Không thể xóa {file_path}: {e}")
    else:
        os.makedirs(LETTER_FOLDER, exist_ok=True)
    
    print(f"✅ Thư mục letters đã sạch: {LETTER_FOLDER}")

    # ---- Step 5: Xử lý từng ký tự ----
    for idx, (x, y, w, h, cnt) in enumerate(filtered_contours):
        # Crop từ ảnh threshold
        crop = th[y:y+h, x:x+w]

        # Tiền xử lý
        processed = preprocessing_for_prediction(crop)

        # Lưu ảnh để hiển thị
        save_path = os.path.join(LETTER_FOLDER, f"letter_{idx}.png")
        cv2.imwrite(save_path, (processed.squeeze() * 255))
        print(f"💾 Đã lưu: {save_path}")

        # ✅ FIX: Trả về tên file thôi, không có "letters/"
        letters.append(f"letter_{idx}.png")

        # Dự đoán
        inp = np.expand_dims(processed, axis=0)
        pred = model.predict(inp, verbose=0)
        class_index = np.argmax(pred)
        confidence = pred[0][class_index]
        
        # Thêm confidence để debug
        predictions.append({
            "label": LABELS[class_index],
            "confidence": float(confidence)
        })

    print(f"📊 Tổng số ký tự phát hiện: {len(letters)}")
    print(f"🎯 Dự đoán: {[p['label'] if isinstance(p, dict) else p for p in predictions]}")
    
    return jsonify({
        "steps": steps,
        "letters": letters,
        "predictions": predictions
    })

# --- RUN APP ---
if __name__ == "__main__":
    app.run(debug=True)