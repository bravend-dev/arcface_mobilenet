import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from mobilenet_embedding import ArcFaceConfig, MobileNetV2WithArcFace
from PIL import Image
import torch.nn.functional as F

from PIL import Image, ImageOps
import numpy as np
import cv2

def crop_largest_face(pil_img: Image.Image, expand: float = 0.3, min_size: int = 64) -> Image.Image:
    """
    Cắt khuôn mặt có diện tích lớn nhất từ ảnh PIL.
    - expand: tỉ lệ mở rộng bbox mỗi chiều (0.25 = 25%) để đỡ cắt cụt trán/cằm.
    - min_size: bỏ qua các bbox quá nhỏ (pixel).
    - use_face_recognition: True nếu muốn dùng thư viện face_recognition (nếu đã cài).
    Trả về ảnh PIL đã cắt; nếu không tìm thấy mặt thì trả lại ảnh gốc.
    """
    # Sửa hướng ảnh theo EXIF (nếu có)
    pil_img = ImageOps.exif_transpose(pil_img)

    # Dùng OpenCV Haar Cascade
    cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # tải model cascade đi kèm opencv
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # tăng tốc bằng cách resize ảnh (tuỳ chọn)
    scale = 1.0
    max_side = max(cv_img.shape[0], cv_img.shape[1])
    if max_side > 1200:
        scale = 1200.0 / max_side
        cv_img_small = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        cv_img_small = cv_img

    gray = cv2.cvtColor(cv_img_small, cv2.COLOR_BGR2GRAY)
    # tune tham số tùy ảnh của bạn
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # scale ngược bbox về kích thước gốc
    rects = []
    for (x, y, w, h) in faces:
        if scale != 1.0:
            x = int(x / scale); y = int(y / scale); w = int(w / scale); h = int(h / scale)
        rects.append((x, y, w, h))

    if not rects:
        # không thấy mặt → trả ảnh gốc
        return pil_img

    # chọn bbox có diện tích lớn nhất
    rects.sort(key=lambda r: r[2]*r[3], reverse=True)
    x, y, w, h = rects[0]
    if w < min_size or h < min_size:
        return pil_img

    # mở rộng bbox
    cx, cy = x + w/2, y + h/2
    expand_w = w * (1 + expand)
    expand_h = h * (1 + expand)

    # để khung gần vuông hơn (tùy gu thẩm mỹ khi encode face)
    side = int(max(expand_w, expand_h))
    x1 = int(cx - side/2)
    y1 = int(cy - side/2)
    x2 = x1 + side
    y2 = y1 + side

    # giới hạn trong ảnh
    W, H = pil_img.size
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    # cắt
    cropped = pil_img.crop((x1, y1, x2, y2))
    return cropped

def get_casia_val_transforms(img_size=(112, 96)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # 6. normalize về [-1,1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])


class ImageEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, type_list, transform = None):
        self.image_paths = image_paths
        self.labels = labels
        self.type_list = type_list
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = crop_largest_face(img)

        if self.transform:
            img = self.transform(img)
        
        item = {}
        item["pixel_values"] = img
        item["labels"] = torch.tensor(self.labels[idx])
        item["type"] = self.type_list[idx]

        return item
    
def singleton(cls):
    _instances = {}

    def get_instance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return get_instance

@singleton
class Encoder:
    def __init__(self,  device="cuda", batch_size=64, img_size=(112, 96)):
        self.device = device
        self.batch_size = batch_size
        
        # Load pre-trained model
        model_checkpoint = './checkpoint-124000'
        pre_model = MobileNetV2WithArcFace.from_pretrained(model_checkpoint)
        self.model = MobileNetV2WithArcFace.from_pretrained(model_checkpoint)
        self.model.backbone = pre_model.backbone
        self.model.to(self.device)
        self.model.eval()

        # Set transform
        self.transform = get_casia_val_transforms(img_size=img_size)

    def encode_img(self, img):
        """
        Encode a single image and return a 1D embedding of shape (D,).
        Expects self.transform to produce a torch.Tensor in CHW format.
        """
        if self.transform:
            img = self.transform(img)  # -> Tensor [C,H,W]

        # Ensure batch dimension: [1,C,H,W]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img = img.unsqueeze(0)
            elif img.dim() != 4:
                raise ValueError("After transform, expected tensor with 3 dims [C,H,W] or 4 dims [B,C,H,W].")
        else:
            raise TypeError("transform must return a torch.Tensor.")

        with torch.no_grad():
            inputs = img.to(self.device).float()
            outputs = self.model(pixel_values=inputs)          # embeddings: [1, D]
            embedding = F.normalize(outputs.embedding, p=2, dim=1)
            return embedding.cpu().numpy()
        
    def encode_paths(self, image_paths):
        # Dummy labels and type list just to use the dataset
        labels = [0] * len(image_paths)
        type_list = ["query"] * len(image_paths)

        dataset = ImageEmbeddingDataset(image_paths, labels, type_list, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding images"):
                inputs = batch["pixel_values"].to(self.device)
                outputs = self.model(pixel_values=inputs)
                embeddings = F.normalize(outputs.embedding, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings
