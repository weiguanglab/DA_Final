import json
from typing import List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset
import os


class DataLoader:
    @staticmethod
    def load_data() -> Tuple[List[Dict], List[Dict]]:
        item_pool = []
        print(f"Loading metadata from: Amazon-C4/sampled_item_metadata_1M.jsonl")
        with open("Amazon-C4/sampled_item_metadata_1M.jsonl", 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Loading metadata"):
                try:
                    item_pool.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format in line: {line.strip()}")
        print(f"Loaded {len(item_pool)} items")
        dataset = load_dataset('Amazon-C4')['test']
        return item_pool, dataset

    @staticmethod
    def create_metadata_map(item_pool: List[Dict]) -> Dict[str, Dict]:
        print("Creating metadata map...")
        metadata_map = {}
        for item in tqdm(item_pool, desc="Creating map"):
            item_id = item.get('item_id')
            if item_id:
                metadata_map[item_id] = {
                    'metadata': item.get('metadata'),
                    'category': item.get('category')
                }
        print(f"Created map for {len(metadata_map)} items")
        return metadata_map

    @staticmethod
    def filter_and_construct(dataset, metadata_map: Dict[str, Dict]) -> List[Dict]:
        print("Filtering and constructing data...")
        new_list = []

        for data in tqdm(dataset, desc="Processing data"):
            item_id = data['item_id']
            item_info = metadata_map.get(item_id, {'metadata': None, 'category': None})
            metadata = item_info['metadata']

            if metadata is None or len(metadata.split()) < 10:
                continue

            new_entry = {
                'query': data['query'],
                'item_id': item_id,
                'metadata': metadata,
                'category': item_info['category']
            }
            new_list.append(new_entry)

        print(f"Constructed {len(new_list)} valid entries")
        return new_list

    @staticmethod
    def save_List_of_Dict(list_of_dict: List[Dict], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in list_of_dict:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"Saved {len(list_of_dict)} entries to {file_path}")

    @staticmethod
    def load_List_of_Dict(file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data_list = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON format in line {line_num}: {line}")
                        print(f"Error: {e}")
                        continue

        print(f"Loaded {len(data_list)} entries from {file_path}")
        return data_list


    @staticmethod
    def save_dict(dict: Dict, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False)

        print(f"Saved dictionary to {file_path}")

    @staticmethod
    def load_dict(file_path: str) -> Dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded dictionary from {file_path}")
        return data
