import numpy as np
import torch
from typing import List, Dict, Tuple
import json
import math
import os


class PerformanceAnalyzer:
    @staticmethod
    def load_data(classifier_results_path: str, query_embeddings_path: str, item_embeddings_path: str) -> Tuple[List[Dict], Dict, Dict]:
        """Load data from saved files"""
        print("Loading data...")

        # Load classifier results
        classifier_results = []
        with open(classifier_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                classifier_results.append(json.loads(line.strip()))
        print(f"Loaded {len(classifier_results)} classifier results")

        # Load query embeddings
        with open(query_embeddings_path, 'r', encoding='utf-8') as f:
            query_embeddings = json.load(f)
        print(f"Loaded {len(query_embeddings)} query embeddings")

        # Load item embeddings
        with open(item_embeddings_path, 'r', encoding='utf-8') as f:
            item_embeddings = json.load(f)
        print(f"Loaded {len(item_embeddings)} categories with item embeddings")

        return classifier_results, query_embeddings, item_embeddings

    @staticmethod
    def save_detailed_results(detailed_results: List[Dict], save_path: str):
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Detailed results saved to: {save_path}")

    @staticmethod
    def prepare_embeddings(query_embedding_dict: Dict, item_embedding_dict: Dict) -> Tuple[Dict, Dict]:
        print("Preparing embeddings for similarity computation...")

        query_embeddings_tensor_dict = {}
        for query, embedding in query_embedding_dict.items():
            query_embeddings_tensor_dict[query] = torch.tensor(embedding, dtype=torch.float32)

        category_embeddings_dict = {}
        for category, items in item_embedding_dict.items():
            if items:
                embeddings = []
                item_ids = []
                for item in items:
                    embeddings.append(item['embedding'])
                    item_ids.append(item['item_id'])

                category_embeddings_dict[category] = {
                    'embeddings': torch.tensor(embeddings, dtype=torch.float32),
                    'item_ids': item_ids
                }

        print(f"Prepared {len(query_embeddings_tensor_dict)} query embeddings")
        print(f"Prepared {len(category_embeddings_dict)} categories")
        for category, data in category_embeddings_dict.items():
            print(f"  {category}: {len(data['item_ids'])} items")

        return query_embeddings_tensor_dict, category_embeddings_dict

    @staticmethod
    def compute_similarity(query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        scores = (query_norm @ doc_norm.T).squeeze(0) * 100
        return scores

    @staticmethod
    def calculate_recall_at_k(retrieved_items: List[str], relevant_item: str, k: int) -> float:
        if k > len(retrieved_items):
            k = len(retrieved_items)

        top_k_items = retrieved_items[:k]
        return 1.0 if relevant_item in top_k_items else 0.0

    @staticmethod
    def calculate_ndcg_at_k(retrieved_items: List[str], relevant_item: str, k: int) -> float:
        if k > len(retrieved_items):
            k = len(retrieved_items)

        top_k_items = retrieved_items[:k]

        dcg = 0.0
        for i, item_id in enumerate(top_k_items):
            if item_id == relevant_item:
                dcg += 1.0 / math.log2(i + 2)
                break

        idcg = 1.0 / math.log2(2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg

    @staticmethod
    def get_top_k_items_global(query: str,
                              query_embeddings_dict: Dict,
                              category_embeddings_dict: Dict,
                              k: int = 200) -> Tuple[List[str], List[float]]:
        """Global search across all categories without filtering"""
        if query not in query_embeddings_dict:
            return [], []

        query_embedding = query_embeddings_dict[query]

        all_similarities = []
        all_item_ids = []

        # Search across all categories
        for category, category_data in category_embeddings_dict.items():
            category_embeddings = category_data['embeddings']
            category_item_ids = category_data['item_ids']

            similarities = PerformanceAnalyzer.compute_similarity(query_embedding, category_embeddings)

            for i, (sim_score, item_id) in enumerate(zip(similarities.tolist(), category_item_ids)):
                all_similarities.append(sim_score)
                all_item_ids.append(item_id)

        if not all_similarities:
            return [], []

        similarities_tensor = torch.tensor(all_similarities)

        k_actual = min(k, len(all_similarities))
        top_k_values, top_k_indices = torch.topk(similarities_tensor, k=k_actual)

        top_k_items = [all_item_ids[idx] for idx in top_k_indices]
        top_k_scores = top_k_values.tolist()

        if len(top_k_scores) < k:
            top_k_scores.extend([0.0] * (k - len(top_k_scores)))

        return top_k_items, top_k_scores

    @staticmethod
    def get_top_k_items_from_predicted_categories(query: str,
                                                 query_embeddings_dict: Dict,
                                                 category_embeddings_dict: Dict,
                                                 pred_categories: List[str],
                                                 k: int = 200) -> Tuple[List[str], List[float]]:
        if query not in query_embeddings_dict:
            return [], []

        query_embedding = query_embeddings_dict[query]

        all_similarities = []
        all_item_ids = []

        for category in pred_categories:
            if category not in category_embeddings_dict:
                continue

            category_data = category_embeddings_dict[category]
            category_embeddings = category_data['embeddings']
            category_item_ids = category_data['item_ids']

            similarities = PerformanceAnalyzer.compute_similarity(query_embedding, category_embeddings)

            for i, (sim_score, item_id) in enumerate(zip(similarities.tolist(), category_item_ids)):
                all_similarities.append(sim_score)
                all_item_ids.append(item_id)

        if not all_similarities:
            return [], []

        similarities_tensor = torch.tensor(all_similarities)

        k_actual = min(k, len(all_similarities))
        top_k_values, top_k_indices = torch.topk(similarities_tensor, k=k_actual)

        top_k_items = [all_item_ids[idx] for idx in top_k_indices]
        top_k_scores = top_k_values.tolist()

        if len(top_k_scores) < k:
            top_k_scores.extend([0.0] * (k - len(top_k_scores)))

        return top_k_items, top_k_scores

    @staticmethod
    def analyze_performance(classifier_results: List[Dict],
                          query_embedding_dict: Dict,
                          item_embedding_dict: Dict,
                          k_values: List[int] = [10, 50, 100, 200],
                          use_category_filtering: bool = True) -> Dict:
        print(f"Starting performance analysis (category filtering: {use_category_filtering})...")

        query_embeddings_tensor_dict, category_embeddings_dict = PerformanceAnalyzer.prepare_embeddings(
            query_embedding_dict, item_embedding_dict
        )

        results = {
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'total_queries': 0,
            'successful_queries': 0,
            'use_category_filtering': use_category_filtering
        }

        detailed_results = []

        print(f"Analyzing {len(classifier_results)} queries...")

        for i, result in enumerate(classifier_results):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(classifier_results)} queries processed")

            query = result['query']
            real_item_id = result['real_item_id']
            real_category = result['real_category']
            pred_categories = result['pred_categories']

            query_with_prefix = "query: " + query

            results['total_queries'] += 1

            if use_category_filtering:
                top_items, top_scores = PerformanceAnalyzer.get_top_k_items_from_predicted_categories(
                    query_with_prefix,
                    query_embeddings_tensor_dict,
                    category_embeddings_dict,
                    pred_categories,
                    k=200
                )
            else:
                top_items, top_scores = PerformanceAnalyzer.get_top_k_items_global(
                    query_with_prefix,
                    query_embeddings_tensor_dict,
                    category_embeddings_dict,
                    k=200
                )

            if not top_items:
                detailed_result = {
                    'correct_item_rank': -1,
                    'category_in_prediction': False,
                    'top_200_scores': []
                }
                detailed_results.append(detailed_result)
                continue

            results['successful_queries'] += 1

            correct_item_rank = -1
            if real_item_id in top_items:
                correct_item_rank = top_items.index(real_item_id) + 1

            category_in_prediction = real_category in pred_categories

            detailed_result = {
                'correct_item_rank': correct_item_rank,
                'category_in_prediction': category_in_prediction,
                'top_200_scores': top_scores
            }
            detailed_results.append(detailed_result)

            for k in k_values:
                recall_k = PerformanceAnalyzer.calculate_recall_at_k(top_items, real_item_id, k)
                results['recall'][k].append(recall_k)

                ndcg_k = PerformanceAnalyzer.calculate_ndcg_at_k(top_items, real_item_id, k)
                results['ndcg'][k].append(ndcg_k)

        filtering_suffix = "" if use_category_filtering else "_without_classifier"
        PerformanceAnalyzer.save_detailed_results(
            detailed_results,
            f"cache/analysis/detailed_results{filtering_suffix}.jsonl"
        )

        performance_summary = {
            'avg_recall': {},
            'avg_ndcg': {},
            'total_queries': results['total_queries'],
            'successful_queries': results['successful_queries'],
            'success_rate': results['successful_queries'] / results['total_queries'] if results['total_queries'] > 0 else 0,
            'use_category_filtering': use_category_filtering,
            'method': 'category_filtered' if use_category_filtering else 'global_search'
        }

        for k in k_values:
            if results['recall'][k]:
                performance_summary['avg_recall'][f'recall@{k}'] = np.mean(results['recall'][k])
                performance_summary['avg_ndcg'][f'ndcg@{k}'] = np.mean(results['ndcg'][k])
            else:
                performance_summary['avg_recall'][f'recall@{k}'] = 0.0
                performance_summary['avg_ndcg'][f'ndcg@{k}'] = 0.0

        return performance_summary

    @staticmethod
    def print_results(performance_summary: Dict):
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS RESULTS")
        print("="*60)

        print(f"Search method: {performance_summary.get('method', 'unknown')}")
        print(f"Category filtering: {performance_summary.get('use_category_filtering', 'unknown')}")
        print(f"Total queries: {performance_summary['total_queries']}")
        print(f"Successful queries: {performance_summary['successful_queries']}")
        print(f"Success rate: {performance_summary['success_rate']:.4f}")

        print("\nRecall@k:")
        for metric, value in performance_summary['avg_recall'].items():
            print(f"  {metric}: {value:.4f}")

        print("\nNDCG@k:")
        for metric, value in performance_summary['avg_ndcg'].items():
            print(f"  {metric}: {value:.4f}")

        print("="*60)

    @staticmethod
    def save_results(performance_summary: Dict, save_path: str):
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(performance_summary, f, indent=4, ensure_ascii=False)

        print(f"Results saved to: {save_path}")
