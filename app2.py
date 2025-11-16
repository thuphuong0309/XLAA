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
#LABELS = ['Y', 'Z', 'X', 'W', 'T', 'U', 'triangle', 'R', 'square', 'V', 'Q', 'P', 'O', 'L', 'N', 'M', 'I', 'K', 'J', 'G', 'C', 'E', 'B', 'A', 'F', 'circle', '9', '7', '6', '4', '5', '3', '2', '0', '1', '8', 'D', 'H']
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
    pad_h = int(h * 0.4)  # 20% chiều cao  # mới sửa chỗ này
    pad_w = int(w * 0.4)  # 20% chiều rộng
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

    # ---- Nhận tham số threshold method ----
    threshold_method = "combined"  # Default
    
    # ---- Nhận ảnh ----
    if "image" in request.files:
        file = request.files["image"]
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        threshold_method = request.form.get("threshold_method", "combined")
    else:
        data = request.get_json()
        if data is None or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        base64_str = data["image"].split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        threshold_method = data.get("threshold_method", "combined")
    
    print(f"🎯 Phương pháp threshold: {threshold_method}")

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

    # ---- Step 3: Threshold (thử 2 phương pháp) ----
    # Phương pháp 1: Otsu (tốt cho ảnh đồng đều)
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{STATIC_FOLDER}/step_thresh_otsu.png", th_otsu)
    steps.append({"title": "Threshold (Otsu)", "file": "step_thresh_otsu.png"})
    
    # Phương pháp 2: Adaptive (tốt cho ánh sáng không đều)
    th_adaptive = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=21,  # Kích thước vùng xét (càng lớn càng mượt)
        C=10           # Hằng số trừ đi (điều chỉnh độ nhạy)
    )
    cv2.imwrite(f"{STATIC_FOLDER}/step_thresh_adaptive.png", th_adaptive)
    steps.append({"title": "Threshold (Adaptive)", "file": "step_thresh_adaptive.png"})
    
    #Phương pháp 3: Kết hợp 2 phương pháp: lấy giao (AND)
    th_combined = cv2.bitwise_or(th_otsu, th_adaptive)
    cv2.imwrite(f"{STATIC_FOLDER}/step_thresh_combined.png", th_combined)
    steps.append({"title": "Threshold (Combined)", "file": "step_thresh_combined.png"})
    
    # ---- Step 3.5: Morphology để làm sạch ----
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Closing: lấp đầy lỗ trong ký tự
    th_morphed = cv2.morphologyEx(th_combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Opening: loại bỏ nhiễu nhỏ
    th_morphed = cv2.morphologyEx(th_morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(f"{STATIC_FOLDER}/step_morphology.png", th_morphed)
    steps.append({"title": "Morphology (Clean)", "file": "step_morphology.png"})
    
    # ---- Step 3.6: Canny Edge Detection ----
    # Áp dụng Canny trên ảnh blur
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)
    # Dilate để nối các cạnh gần nhau
    kernel_canny = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_canny, iterations=2)
    # Closing để lấp đầy
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_canny, iterations=2)
    cv2.imwrite(f"{STATIC_FOLDER}/step_canny.png", edges_closed)
    steps.append({"title": "Canny Edge Detection", "file": "step_canny.png"})
    
    # Chọn ảnh threshold để PHÁT HIỆN contours
    if threshold_method == "otsu":
        th_detect = th_otsu
        print("✅ Dùng Otsu để phát hiện")
    elif threshold_method == "adaptive":
        th_detect = th_adaptive
        print("✅ Dùng Adaptive để phát hiện")
    elif threshold_method == "canny":
        th_detect = edges_closed
        print("✅ Dùng Canny Edge để phát hiện")
    else:  # combined
        th_detect = th_morphed
        print("✅ Dùng Combined để phát hiện")
    
    # QUAN TRỌNG: Luôn dùng Otsu để crop ký tự (cho model dự đoán)
    th_for_crop = th_otsu
    print("✅ Dùng Otsu để crop ký tự (model dự đoán tốt hơn)")

    # ---- Step 4: Tách ký tự (dùng th_detect) ----
    contours, _ = cv2.findContours(th_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours quá nhỏ hoặc quá lớn
    filtered_contours = []
    img_area = th_detect.shape[0] * th_detect.shape[1]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Bỏ qua contour quá nhỏ (<1% ảnh) hoặc quá lớn (>80% ảnh)
        if area > img_area * 0.01 and area < img_area * 0.8:
            if w > 10 and h > 10:  # Kích thước tối thiểu
                filtered_contours.append((x, y, w, h, cnt))
    
    # ---- Sắp xếp theo dòng (trên → dưới) rồi trái → phải ----
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
    
    # Lưu ảnh threshold được dùng để phát hiện
    cv2.imwrite(f"{STATIC_FOLDER}/step_thresh_used.png", th_detect)
    steps.append({"title": f"Threshold Used ({threshold_method.upper()})", "file": "step_thresh_used.png"})

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
        # Crop từ ảnh Otsu (cho model dự đoán tốt hơn)
        crop = th_for_crop[y:y+h, x:x+w]
        # crop = cv_img[y:y+h, x:x+w]

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