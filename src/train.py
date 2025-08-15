from transformers import TrainingArguments
from data.ic_dataset import (
    ImageClassificationDataset,
    load_wc_data
)
from data.is_dataset import (
    ImageEmbeddingDataset,
    load_lfw_data_with_query
)
from data.augment import (
    get_casia_train_transforms,
    get_casia_val_transforms
)
from models.mobilenet_embedding import MobileNetV2WithArcFace, ArcFaceConfig
from trainers.arcface_trainer import ArcFaceTrainer

from transformers import TrainingArguments, get_scheduler
from torch import nn
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

train_image_paths, train_labels, train_label2index = load_wc_data()
# train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
#     train_image_paths, train_labels, test_size=0.05, random_state=42, stratify=train_labels)

num_train_class = len(train_label2index)
test_image_paths, test_labels, test_type_list, test_label2index = load_lfw_data_with_query()

model_config = ArcFaceConfig(num_classes=num_train_class)

train_dataset = ImageClassificationDataset(
    train_image_paths, train_labels,
    transform=get_casia_train_transforms()
)
# eval_dataset = ImageClassificationDataset(
#     test_image_paths, test_labels,
#     transform=get_casia_val_transforms()
# )
eval_dataset = ImageEmbeddingDataset(
    test_image_paths, test_labels, test_type_list,
    transform=get_casia_val_transforms()
)

pre_model = MobileNetV2WithArcFace.from_pretrained('/home/mfite/Workspaces/dungnd/face_recognition/vibe_code/checkpoints/best/checkpoint-124000')
# Load model
model = MobileNetV2WithArcFace(model_config)
model.backbone = pre_model.backbone

# ckpt = torch.load('/home/mfite/Workspaces/dungnd/face_recognition/068.ckpt')
# model.backbone.load_state_dict(ckpt['net_state_dict'])

training_args = TrainingArguments(
    output_dir="./checkpoints",
    
    # đánh giá và lưu theo số bước
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    
    # tự động nạp lại model tốt nhất khi kết thúc
    load_best_model_at_end=True,
    metric_for_best_model="top_1",
    greater_is_better=True,
    
    # load_best_model_at_end=True,
    # metric_for_best_model="loss",
    # greater_is_better=False,
    
    # chỉ giữ lại tối đa 3 checkpoint
    save_total_limit=3,
    
    # các tham số khác
    per_device_train_batch_size=256,
    per_device_eval_batch_size=128,
    num_train_epochs=70,
    logging_dir="./logs",
    logging_steps=100,
    
    do_train=False,        # không train
    do_eval=True,          # bật evaluate
)

# define optimizers
ignored_params = list(map(id, model.backbone.linear1.parameters()))
ignored_params += list(map(id, model.loss_func.weight))
prelu_params_id = []
prelu_params = []
for m in model.backbone.modules():
    if isinstance(m, nn.PReLU):
        ignored_params += list(map(id, m.parameters()))
        prelu_params += m.parameters()
base_params = filter(lambda p: id(p) not in ignored_params, model.backbone.parameters())

# optimizer_ft = optim.SGD([
#     {'params': base_params, 'weight_decay': 4e-5},
#     {'params': model.backbone.linear1.parameters(), 'weight_decay': 4e-4},
#     {'params': model.loss_func.weight, 'weight_decay': 4e-4},
#     {'params': prelu_params, 'weight_decay': 0.0}
# ], lr=0.1, momentum=0.9, nesterov=True)

# optimizer_ft = AdamW(
#     [
#         {'params': base_params, 'weight_decay': 4e-5},
#         {'params': model.backbone.linear1.parameters(), 'weight_decay': 4e-4},
#         {'params': model.loss_func.weight, 'weight_decay': 4e-4},
#         {'params': prelu_params, 'weight_decay': 0.0}
#     ],
#     lr=2e-5,
#     betas=(0.9, 0.999)  # giá trị mặc định tốt, có thể tinh chỉnh nếu cần
# )

optimizer_ft = AdamW(
    [
        {'params': base_params,              'lr': 2e-5,  'weight_decay': 4e-5},
        {'params': model.backbone.linear1.parameters(), 'lr': 1e-3, 'weight_decay': 4e-4},
        {'params': model.loss_func.weight,   'lr': 1e-3,  'weight_decay': 4e-4},
        {'params': prelu_params,             'lr': 1e-3,  'weight_decay': 0.0}
    ]
)
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)

num_training_steps = (
    len(train_dataset) // training_args.per_device_train_batch_size
    // training_args.gradient_accumulation_steps
    * training_args.num_train_epochs
)

warmup_steps = int(0.1 * num_training_steps)

# scheduler = get_scheduler(
#     name="linear",
#     optimizer=optimizer_ft,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=num_training_steps,
# )

from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer_ft,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5
)


# Trainer
trainer = ArcFaceTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer_ft, scheduler),
)
# trainer.train()
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)