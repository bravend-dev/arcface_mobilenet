from PIL import Image
import pandas as pd
import torch
import os

def load_wc_data():
    image_folder = '/home/mfite/Workspaces/dungnd/face_recognition/data/casia-webface/casia-webface'
    # image_folder = '/home/mfite/Workspaces/dungnd/face_recognition/data/lfw-cropped-faces/lfw_cut'
    image_paths = []
    labels = []

    # Duyệt qua các thư mục (mỗi người là một class)
    for person_id in sorted(os.listdir(image_folder)):
        person_path = os.path.join(image_folder, person_id)
        if not os.path.isdir(person_path):
            continue
        for filename in os.listdir(person_path):
            if not filename.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(person_path, filename)
            image_paths.append(img_path)
            labels.append(person_id)  # gán label là tên người

    assert len(labels) == len(image_paths)

    print(f'Tổng số mẫu ban đầu: {len(labels)}')
    print(f'Tổng số lớp ban đầu: {len(set(labels))}')

    df = pd.DataFrame({
        'image_path': image_paths,
        'label_name': labels
    })

    # Lọc bỏ các lớp có ít hơn 2 ảnh
    # df = df.groupby('label_name').filter(lambda x: len(x) >= 2)
    
    # df = df.groupby('label_name')

    # Tạo lại label2idx chỉ từ những lớp còn lại
    unique_labels = sorted(df['label_name'].unique())
    label2idx = {name: idx for idx, name in enumerate(unique_labels)}

    image_paths = df['image_path'].tolist()
    labels = [label2idx[label] for label in df['label_name'].tolist()]

    print(f'\nSau khi lọc:')
    print(f'  Số lượng mẫu: {len(labels)}')
    print(f'  Số lớp hợp lệ: {len(label2idx)}')

    return image_paths, labels, label2idx

class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # inputs = self.processor(images=img, return_tensors="pt")
        # item = {k: v.squeeze(0) for k, v in inputs.items()}

        item = {}
        item["pixel_values"] = img
        item["labels"] = torch.tensor(self.labels[idx])
        return item