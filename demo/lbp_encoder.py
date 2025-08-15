#!/usr/bin/env python3
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)  # Tắt cảnh báo

def get_preprocess_model():
    frontal = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml')
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    return frontal, profile, eye

def detect_face(img, frontal_cascade, profile_cascade):
    """
    Phát hiện khuôn mặt từ ảnh đầu vào, sử dụng cả bộ cascade khuôn mặt chính diện và nghiêng (profile).
    Nếu không phát hiện ở ảnh gốc, thử ảnh lật ngang để bắt được mặt nghiêng hướng ngược lại.

    Tham số:
    - img: ảnh đầu vào (dạng BGR từ OpenCV).
    - frontal_cascade: bộ phát hiện khuôn mặt chính diện (Haar cascade).
    - profile_cascade: bộ phát hiện khuôn mặt nghiêng (nghiêng trái hoặc phải).

    Trả về:
    - ROI (Region of Interest) khuôn mặt đã chuyển sang ảnh xám (grayscale), hoặc None nếu không tìm thấy
      hoặc nếu khuôn mặt nhỏ hơn 40% diện tích ảnh.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = img.shape[0] * img.shape[1]
    min_face_area = 0.2 * img_area  # 40% diện tích ảnh

    # --- Giai đoạn 1: Thử phát hiện trên ảnh gốc ---
    for cascade in (frontal_cascade, profile_cascade):
        dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(dets) > 0:
            # Chọn box lớn nhất
            x, y, w, h = max(dets, key=lambda b: b[2] * b[3])
            face_area = w * h
            if face_area >= min_face_area:
                return gray[y:y+h, x:x+w]

    # --- Giai đoạn 2: Thử với ảnh lật ngang ---
    flipped = cv2.flip(gray, 1)

    for cascade in (frontal_cascade, profile_cascade):
        dets = cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5)

        if len(dets) > 0:
            x, y, w, h = max(dets, key=lambda b: b[2] * b[3])
            face_area = w * h
            if face_area >= min_face_area:
                face_roi = flipped[y:y+h, x:x+w]
                return cv2.flip(face_roi, 1)

    return None

def align_face(face_img, eye_cascade, output_size=(128, 128)):
    """
    Căn chỉnh khuôn mặt bằng cách phát hiện vị trí 2 mắt và xoay ảnh sao cho 2 mắt nằm ngang.
    
    Tham số:
    - face_img: ảnh đầu vào của khuôn mặt (ảnh xám - grayscale).
    - eye_cascade: bộ phát hiện mắt (Haar cascade classifier).
    - output_size: kích thước ảnh đầu ra sau khi căn chỉnh (default: 128x128).
    
    Trả về:
    - ảnh khuôn mặt đã được căn chỉnh và resize về kích thước chuẩn.
    """

    # Ảnh đầu vào được giả định là ảnh xám (grayscale)
    gray = face_img
    h, w = gray.shape[:2]  # Lấy chiều cao và chiều rộng ảnh

    # Phát hiện các vùng chứa mắt trên toàn ảnh
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Nếu phát hiện ít hơn 2 mắt, thử lại chỉ trên nửa trên ảnh (vì mắt thường ở nửa trên khuôn mặt)
    if len(eyes) < 2:
        roi = gray[:h//2, :]  # Vùng nửa trên ảnh
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5)
        # Điều chỉnh lại tọa độ Y vì vùng ROI bắt đầu từ y=0
        eyes = [(x, y, ew, eh) for (x, y, ew, eh) in eyes]

    # Nếu phát hiện được ít nhất 2 mắt:
    if len(eyes) >= 2:
        # Chọn 2 mắt có tọa độ y nhỏ nhất (gần phía trên nhất) – giả định là mắt
        eyes = sorted(eyes, key=lambda e: e[1])[:2]
        # Tính tọa độ trung tâm của mỗi mắt
        centers = [(int(x + ew / 2), int(y + eh / 2)) for x, y, ew, eh in eyes]
    else:
        # Nếu không phát hiện được 2 mắt, sử dụng ước lượng vị trí 2 mắt tương đối
        centers = [
            (int(0.3 * w), int(0.35 * h)),  # Mắt trái giả định
            (int(0.7 * w), int(0.35 * h))   # Mắt phải giả định
        ]

    # Phân loại lại mắt trái và phải dựa trên vị trí X
    left_eye, right_eye = sorted(centers, key=lambda c: c[0])

    # Tính toán chênh lệch tọa độ giữa 2 mắt
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    # Tính góc lệch giữa hai mắt so với đường ngang (theo đơn vị độ)
    angle = math.degrees(math.atan2(dy, dx))

    # Tính tâm xoay là trung điểm giữa hai mắt
    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2)
    )

    # Tạo ma trận xoay (quay quanh tâm mắt, góc "angle", không scale)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Áp dụng xoay ảnh theo ma trận M
    aligned = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC)

    # Resize ảnh đã căn chỉnh về kích thước đầu ra chuẩn (mặc định 128x128)
    return cv2.resize(aligned, output_size)

def extract_lbp_hist(image, radius=3, neighbors=16, grid_x=8, grid_y=8):
    """
    Trích xuất histogram LBP cho từng ô nhỏ (cell) trên ảnh và ghép lại thành vector đặc trưng.

    Tham số:
    - image: ảnh grayscale đầu vào (2D numpy array).
    - radius: bán kính tính toán LBP (mặc định: 3).
    - neighbors: số lượng điểm lân cận dùng trong tính LBP (mặc định: 8).
    - grid_x, grid_y: chia ảnh thành lưới grid_x x grid_y ô.

    Trả về:
    - Vector đặc trưng (numpy array) gồm các histogram LBP của từng ô nối lại.
    """

    H, W = image.shape  # Kích thước ảnh
    cell_h, cell_w = H // grid_y, W // grid_x  # Kích thước mỗi ô lưới (cell)

    features = []  # Danh sách lưu histogram từng ô

    # Duyệt qua từng ô lưới trong ảnh
    for i in range(grid_y):          # Theo chiều dọc
        for j in range(grid_x):      # Theo chiều ngang

            # Cắt ra ô ảnh nhỏ (cell)
            cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            h_c, w_c = cell.shape  # Kích thước ô hiện tại

            # Khởi tạo ma trận LBP cho ô (giảm biên 2*radius để tránh out-of-bounds)
            lbp = np.zeros((h_c - 2*radius, w_c - 2*radius), dtype=np.uint8)

            # Tính toán LBP bằng cách so sánh với các điểm lân cận
            for n in range(neighbors):
                # Góc quay của điểm lân cận thứ n
                theta = 2 * np.pi * n / neighbors
                dx = int(round(radius * np.cos(theta)))   # Dịch ngang
                dy = int(round(-radius * np.sin(theta)))  # Dịch dọc (âm vì trục Y ngược)

                # Lấy ảnh điểm lân cận và điểm trung tâm để so sánh
                neighbor = cell[radius+dy:h_c-radius+dy, radius+dx:w_c-radius+dx]
                center   = cell[radius:h_c-radius, radius:w_c-radius]

                # So sánh: nếu điểm lân cận >= điểm trung tâm, gán bit = 1
                # Bit này được dịch trái theo thứ tự n (0..neighbors-1)
                lbp += ((neighbor >= center) << n).astype(np.uint8)

            # Tính histogram LBP cho ô hiện tại
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

            # Chuẩn hóa histogram (tổng = 1)
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-6)  # Cộng epsilon nhỏ để tránh chia 0

            # Thêm histogram vào danh sách đặc trưng
            features.append(hist)

    # Ghép tất cả histogram từ các ô thành vector đặc trưng cuối cùng
    return np.hstack(features)

def compute_features(image_paths, frontal_cascade, profile_cascade, eye_cascade):
    """
    Thực hiện toàn bộ pipeline xử lý khuôn mặt cho một danh sách ảnh:
    - Phát hiện khuôn mặt (frontal + profile)
    - Căn chỉnh khuôn mặt dựa trên mắt
    - Tiền xử lý ảnh (CLAHE)
    - Trích xuất đặc trưng LBP

    Tham số:
    - image_paths: danh sách đường dẫn ảnh (list of strings)
    - frontal_cascade: bộ Haar cascade phát hiện khuôn mặt chính diện
    - profile_cascade: bộ Haar cascade phát hiện khuôn mặt nghiêng
    - eye_cascade: bộ Haar cascade phát hiện mắt (dùng để align)

    Trả về:
    - ma trận đặc trưng (numpy array) có kích thước (n_samples, feature_dim)
    """

    feats = []  # Danh sách lưu đặc trưng từng ảnh

    # Duyệt qua từng ảnh và hiển thị tiến trình bằng tqdm
    for path in tqdm(image_paths, desc="Computing features"):
        img = cv2.imread(path)  # Đọc ảnh

        # 1. Phát hiện khuôn mặt (trả về ROI ảnh xám nếu thành công)
        face = detect_face(img, frontal_cascade, profile_cascade)

        # 2. Nếu không tìm thấy khuôn mặt → dùng ảnh xám gốc
        gray = face if face is not None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Căn chỉnh khuôn mặt dựa trên mắt (nếu có)
        aligned = align_face(gray, eye_cascade)

        # 5. Trích xuất đặc trưng LBP và lưu vào danh sách
        feats.append(extract_lbp_hist(aligned))

    # Chuyển danh sách đặc trưng thành ma trận (n_images × n_features)
    return np.vstack(feats)


def visualize_lbp_image(image, radius=1, neighbors=8):
    """Generate a grayscale visualization of the LBP pattern."""
    h, w = image.shape
    lbp_image = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
    for n in range(neighbors):
        theta = 2 * np.pi * n / neighbors
        dx = int(round(radius * np.cos(theta)))
        dy = int(round(-radius * np.sin(theta)))
        neighbor = image[radius+dy:h-radius+dy, radius+dx:w-radius+dx]
        center   = image[radius:h-radius, radius:w-radius]
        lbp_image += ((neighbor >= center) << n).astype(np.uint8)
    return lbp_image

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)