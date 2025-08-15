from PIL import Image
import pandas as pd
import torch
import os

def load_lfw_data_with_query():
    image_folder = '/home/mfite/Workspaces/dungnd/face_recognition/data/cfpw-dataset/view'
    image_paths = []
    labels = []

    # Duyệt qua các thư mục (mỗi thư mục là một người)
    for person_id in sorted(os.listdir(image_folder)):
        person_path = os.path.join(image_folder, person_id)
        if not os.path.isdir(person_path):
            continue
        for filename in os.listdir(person_path):
            if not filename.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(person_path, filename)
            image_paths.append(img_path)
            labels.append(person_id)

    assert len(labels) == len(image_paths)

    print(f'Tổng số mẫu ban đầu: {len(labels)}')
    print(f'Tổng số lớp ban đầu: {len(set(labels))}')

    # Tạo DataFrame và lọc bỏ các lớp có ít hơn 2 ảnh
    df = pd.DataFrame({'image_path': image_paths, 'label_name': labels})
    df = df.groupby('label_name').filter(lambda x: len(x) >= 2)

    # Gán label dạng số
    unique_labels = sorted(df['label_name'].unique())
    label2idx = {name: idx for idx, name in enumerate(unique_labels)}
    df['label'] = df['label_name'].map(label2idx)

    collection_paths = []
    collection_labels = []
    query_paths = []
    query_labels = []

    # Với mỗi class, chọn 1 ảnh làm query, phần còn lại là collection
    for label_name, group in df.groupby('label_name'):
        items = group.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        query = items.iloc[0]
        collection = items.iloc[1:]

        query_paths.append(query['image_path'])
        query_labels.append(label2idx[label_name])

        collection_paths.extend(collection['image_path'].tolist())
        collection_labels.extend([label2idx[label_name]] * len(collection))

    print(f'\nSau khi lọc và phân chia:')
    print(f'  Số lượng lớp: {len(label2idx)}')
    print(f'  Query samples: {len(query_paths)}')
    print(f'  Collection samples: {len(collection_paths)}')

    type_list = ['query']*len(query_paths) + ['collection']*len(collection_paths)
    
    return query_paths + collection_paths, query_labels+collection_labels, type_list, label2idx

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
        if self.transform:
            img = self.transform(img)
        
        # inputs = self.processor(images=img, return_tensors="pt")
        # item = {k: v.squeeze(0) for k, v in inputs.items()}
        
        item = {}
        item["pixel_values"] = img
        item["labels"] = torch.tensor(self.labels[idx])
        item["type"] = self.type_list[idx]

        return item
