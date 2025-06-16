import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from typing import List, Dict, Optional
from tqdm import tqdm


class QueryDataset(Dataset):
    def __init__(self, queries: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            query,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class QueryClassifier(nn.Module):
    def __init__(self, num_categories: int, model_name: str = "bert-base-uncased", dropout: float = 0.2):
        super(QueryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_categories)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits


class EmbeddingModel:
    def __init__(self, model_name: str = "multilingual-e5-small"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"Model loaded on device: {self.device}")

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, texts: List[str], max_length: int = 128, batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []

        with torch.no_grad():
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", total=total_batches, unit="batch"):
                batch_texts = texts[i:i + batch_size]

                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

                batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}

                outputs = self.model(**batch_dict)
                embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


class FinalEvaluator:
    @staticmethod
    def compute_similarity(query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        scores = (query_embedding @ doc_embeddings.T) * 100
        return scores


def create_data_loaders(train_queries: List[str],
                        train_labels: List[int],
                        test_queries: List[str],
                        test_labels: List[int],
                        tokenizer,
                        batch_size: int = 32,
                        max_len: int = 128) -> tuple:
    train_dataset = QueryDataset(train_queries, train_labels, tokenizer, max_len)
    test_dataset = QueryDataset(test_queries, test_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    print(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully")
    return tokenizer
