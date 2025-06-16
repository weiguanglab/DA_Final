import random
from typing import List, Dict, Tuple


class DataConstructor:
    @staticmethod
    def split_train_test(data_list: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        print("Splitting train/test data...")
        random.shuffle(data_list)
        split_idx = int(train_ratio * len(data_list))
        train_pool = data_list[:split_idx]
        test_pool = data_list[split_idx:]
        print(f"Train: {len(train_pool)}, Test: {len(test_pool)}")
        return train_pool, test_pool

    @staticmethod
    def prepare_classification_data(train_pool: List[Dict], test_pool: List[Dict]) -> Tuple[List[str], List[int], List[str], List[int], Dict, Dict]:
        print("Preparing classification data...")

        train_valid_items = [item for item in train_pool if item.get('metadata') and item.get('category')]
        train_queries = [item['metadata'] for item in train_valid_items]
        train_categories = [item['category'] for item in train_valid_items]

        test_valid_items = [item for item in test_pool if item.get('metadata') and item.get('category')]
        test_queries = [item['metadata'] for item in test_valid_items]
        test_categories = [item['category'] for item in test_valid_items]

        all_categories = set(train_categories + test_categories)
        unique_categories = sorted(list(all_categories))
        category_to_idx = {category: idx for idx, category in enumerate(unique_categories)}
        idx_to_category = {idx: category for category, idx in category_to_idx.items()}

        train_labels = [category_to_idx[cat] for cat in train_categories]
        test_labels = [category_to_idx[cat] for cat in test_categories]

        from collections import Counter
        train_category_dist = Counter(train_categories)
        test_category_dist = Counter(test_categories)

        print(f"Train: {len(train_queries)} samples")
        print(f"Test: {len(test_queries)} samples")
        print(f"Total categories: {len(unique_categories)}")
        print(f"Categories: {unique_categories}")

        print("\nCategory distribution:")
        for category in unique_categories:
            train_count = train_category_dist.get(category, 0)
            test_count = test_category_dist.get(category, 0)
            print(f"  {category}: train={train_count}, test={test_count}")

        return train_queries, train_labels, test_queries, test_labels, category_to_idx, idx_to_category

    @staticmethod
    def get_item_list(filtered_data: List[Dict]):
        item_dict = {}

        for entry in filtered_data:
            item_id = entry.get('item_id')
            if item_id and item_id not in item_dict:
                item_dict[item_id] = {
                    'item_id': item_id,
                    'category': entry.get('category'),
                    'metadata': entry.get('metadata')
                }

        item_list = list(item_dict.values())
        print(f"Extracted {len(item_list)} unique items from {len(filtered_data)} entries")
        return item_list

    @staticmethod
    def get_item_embedding_dict_with_category(item_list, item_embeddings):
        item_embedding_dict = {}
        if len(item_list) != item_embeddings.shape[0]:
            print(f"Warning: item_list length ({len(item_list)}) doesn't match embeddings shape ({item_embeddings.shape[0]})")
            min_len = min(len(item_list), item_embeddings.shape[0])
            item_list = item_list[:min_len]
            item_embeddings = item_embeddings[:min_len]

        for i, item in enumerate(item_list):
            category = item.get('category')
            item_id = item.get('item_id')

            if category not in item_embedding_dict:
                item_embedding_dict[category] = []

            embedding = item_embeddings[i].tolist()
            item_embedding_dict[category].append({
                'item_id': item_id,
                'embedding': embedding
            })

        total_items = sum(len(v) for v in item_embedding_dict.values())
        print(f"Created embedding dictionary for {len(item_embedding_dict)} categories with {total_items} items")

        for category, items in item_embedding_dict.items():
            print(f"  {category}: {len(items)} items")

        return item_embedding_dict

    @staticmethod
    def get_query_embedding_dict(query_list, query_embeddings):
        query_embedding_dict = {}

        if len(query_list) != query_embeddings.shape[0]:
            print(f"Warning: query_list length ({len(query_list)}) doesn't match embeddings shape ({query_embeddings.shape[0]})")
            min_len = min(len(query_list), query_embeddings.shape[0])
            query_list = query_list[:min_len]
            query_embeddings = query_embeddings[:min_len]

        for i, query in enumerate(query_list):
            embedding = query_embeddings[i].tolist()
            query_embedding_dict[query] = embedding

        print(f"Created embedding dictionary for {len(query_embedding_dict)} queries")
        print(f"Embedding dimension: {len(query_embedding_dict[next(iter(query_embedding_dict))]) if query_embedding_dict else 'N/A'}")

        return query_embedding_dict