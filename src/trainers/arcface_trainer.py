from transformers import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import faiss
from utils.metrics import (
    compute_map_at_k,
    compute_precision_at_k,
    compute_topk_accuracy
)
import torch.nn.functional as F

class ArcFaceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_list = [1, 2, 3, 5, 10]

    def evaluate(self, eval_dataset=None, **kwargs):
        
        orig_metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_embeddings = []
        all_labels = []
        all_types = []

        for batch in tqdm(dataloader, desc="Embedding"):
            # Di chuyển batch lên device
            batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                # Giả sử model trả về embedding là output cuối
                # embeddings = outputs.embedding
                embeddings = F.normalize(outputs.embedding, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())
                all_labels.extend(batch["labels"].cpu().tolist())
                all_types.extend(batch["type"])  # string: "query" hoặc "collection"

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        all_labels = np.array(all_labels)
        all_types = np.array(all_types)

        # Tách collection / query
        collection_mask = all_types == "collection"
        query_mask = all_types == "query"

        collection_embeddings = all_embeddings[collection_mask]
        collection_labels = all_labels[collection_mask]

        query_embeddings = all_embeddings[query_mask]
        query_labels = all_labels[query_mask]

        # FAISS index
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatIP(res, collection_embeddings.shape[1])
        gpu_index.add(collection_embeddings)

        max_k = max(self.top_k_list)
        # Search
        D, I = gpu_index.search(query_embeddings, max_k)  # I: indices

        results = {}

        for k in self.top_k_list:
            topk_I = I[:, :k]

            accuracy = compute_topk_accuracy(topk_I, query_labels, collection_labels)
            precision = compute_precision_at_k(topk_I, query_labels, collection_labels)
            mean_ap = compute_map_at_k(topk_I, query_labels, collection_labels)

            results[f"top_{k}"] = {
                "accuracy": accuracy,
                "precision": precision,
                "mAP": mean_ap
            }

            print(f"Top-{k}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, mAP={mean_ap:.4f}")
        
        results["eval_top_1"] = results[f"top_{1}"]["accuracy"]

        self.model.train()

        merged = {**orig_metrics, **results}

        # --- 4) Khôi phục chế độ train và return ---
        self.model.train()
        
        return merged
    
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):

        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        outputs = model(pixel_values = pixel_values, labels = labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        # Giả sử bạn đã có self.train_dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )