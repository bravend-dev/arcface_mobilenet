from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from transformers.modeling_outputs import ModelOutput
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from mobileFacenet import MobileFacenet, ArcMarginProduct
from typing import Optional
# from pytorch_metric_learning.losses.arcface_loss import ArcFaceLoss

class ArcFaceConfig(PretrainedConfig):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

@dataclass
class ArcFaceOutput(ModelOutput):
    embedding: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None

class MobileNetV2WithArcFace(PreTrainedModel):
    config_class = ArcFaceConfig

    def __init__(self, config: ArcFaceConfig):
        super().__init__(config)
        # self.backbone = AutoModel.from_pretrained(
        #     config.model_name,
        #     trust_remote_code=True,
        #     use_safetensors=True,
        # )
        # self.embedding_dim = 128
        # # projection từ embedding_dim gốc về 128
        # self.projection = nn.Linear(1280, self.embedding_dim, bias=False)
        # self.embedding_bn = nn.BatchNorm1d(self.embedding_dim)

        # # Khởi tạo weight cho projection
        # nn.init.xavier_uniform_(self.projection.weight)

        self.backbone = MobileFacenet()

        self.loss_func = ArcMarginProduct(128, config.num_classes, s=32, m = 0.4)
        # self.loss_func = ArcFaceLoss(num_classes=config.num_classes, embedding_size=128)

    def forward(self, pixel_values: torch.FloatTensor, labels = None, **kwargs) -> ArcFaceOutput:
        # print(f"-----------------> DEV: {pixel_values.shape}")
        # 1) Lấy feature từ backbone
        outputs = self.backbone(pixel_values)
        # x = outputs.pooler_output              # (batch, config.embedding_dim)

        # # 2) Project xuống 128 và BatchNorm
        # x = self.projection(x)                 # (batch, 128)
        # x = self.embedding_bn(x)               # (batch, 128)

        # # 3) L2 normalize để dùng với ArcFace loss
        # embedding = F.normalize(x, p=2, dim=1)  # (batch, 128)

        loss = None
        if labels is not None:
            logits = self.loss_func(outputs, labels)
            loss = F.cross_entropy(logits, labels)
            # loss = self.loss_func(outputs, labels)

        return ArcFaceOutput(embedding=outputs, loss=loss)