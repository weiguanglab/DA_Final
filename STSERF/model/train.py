import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from typing import Optional
from torch.nn.functional import softmax


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: Optional[str] = None,
                 learning_rate: float = 1e-5):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

        self.model = self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        print(f"Trainer initialized on device: {self.device}")

    def train_epoch(self, train_loader, epoch_num: int) -> float:
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch_num}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, train_loader, test_loader=None, num_epochs: int = 3) -> dict:
        print(f"Starting training for {num_epochs} epochs...")

        history = {
            'train_losses': [],
            'epochs': []
        }

        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(train_loader, epoch)

            history['train_losses'].append(avg_loss)
            history['epochs'].append(epoch)

            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

            if test_loader is not None:
                val_loss = self.validate(test_loader)
                print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}")

        print("Training completed!")
        return history

    def validate(self, test_loader) -> float:
        self.model.eval()
        total_loss = 0
        num_batches = len(test_loader)

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / num_batches


    def save_model(self, save_path: str = "cache/model/bert/bert.pth"):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created directory: {save_dir}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, load_path: str = "cache/model/bert/bert.pth"):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Model loaded from: {load_path}")

    def evaluate_topk_accuracy(self, test_loader, k: int = 3) -> float:
        self.model.eval()
        correct_topk = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating Top-{k} Accuracy"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

                _, topk_preds = torch.topk(outputs, k=k, dim=-1)

                for i in range(labels.size(0)):
                    if labels[i].item() in topk_preds[i].tolist():
                        correct_topk += 1
                    total += 1

        topk_accuracy = correct_topk / total if total > 0 else 0.0
        print(f"Top-{k} Accuracy: {topk_accuracy:.4f} ({correct_topk}/{total})")
        return topk_accuracy

    def predict_topk(self, test_loader, test_pool, idx_to_category: dict, k: int = 2) -> list:
        result_list = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Generating Top-{k} Predictions"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probabilities = softmax(logits, dim=-1)

                topk_indices = torch.topk(probabilities, k, dim=-1).indices.cpu().numpy()
                topk_probs = torch.topk(probabilities, k, dim=-1).values.cpu().numpy()

                for i, (topk_idx, topk_prob) in enumerate(zip(topk_indices, topk_probs)):
                    topk_categories = [idx_to_category[idx] for idx in topk_idx]

                    result_list.append({
                        f"pred_categories": topk_categories,
                    })

        for i, result in enumerate(result_list):
            result['query'] = test_pool[i]['query']
            result['real_category'] = test_pool[i]['category']
            result['real_item_id'] = test_pool[i]['item_id']

        print(f"Generated predictions for {len(result_list)} samples")
        return result_list
