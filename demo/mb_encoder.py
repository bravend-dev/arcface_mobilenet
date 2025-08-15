import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from mobilenet_embedding import ArcFaceConfig, MobileNetV2WithArcFace
from PIL import Image
import torch.nn.functional as F

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
        if self.transform:
            img = self.transform(img)
        
        item = {}
        item["pixel_values"] = img
        item["labels"] = torch.tensor(self.labels[idx])
        item["type"] = self.type_list[idx]

        return item

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

    def encode(self, image_paths):
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
