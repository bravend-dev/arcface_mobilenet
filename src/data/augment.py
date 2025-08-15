from torchvision import transforms

def get_casia_train_transforms(img_size=(112, 96)):
    return transforms.Compose([
        # transforms.Resize(img_size),
        # 1. crop random trong khoảng 85–100% diện tích, tỉ lệ ±10%
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1)
        ),
        # 2. flip ngẫu nhiên với p=0.5
        transforms.RandomHorizontalFlip(p=0.5),
        # 3. jitter màu
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4
        ),
        # 4. đôi khi grayscale (với p=0.1)
        transforms.RandomGrayscale(p=0.1),
        # 5. chuyển thành tensor [0,1]
        transforms.ToTensor(),
        # 6. normalize về [-1,1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
        # 7. Erasing (giá trị random), p=0.1
        transforms.RandomErasing(
            p=0.1,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value='random'
        ),
    ])

from torchvision import transforms

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