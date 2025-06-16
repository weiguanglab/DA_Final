from STSERF import *


def main():
    # Load datasets
    train_loader, train_pool, test_loader, test_pool, category_to_idx, idx_to_category = load_datasets()

    # Train the QueryClassifier
    train_QueryClassifier(len(category_to_idx), train_loader, test_loader)

    # Evaluate the QueryClassifier
    evaluate_QueryClassifier(len(category_to_idx), test_loader, test_pool, idx_to_category, 2)

    # Get embeddings
    get_embeddings()

    # Evaluate performance
    analyze_performance(True)
    analyze_performance(False)

    # Visualize performance
    visualize_performance()

    return


def load_datasets():
    print("Loading datasets...")
    itempool, dataset = DataLoader.load_data()
    metadata_map = DataLoader.create_metadata_map(itempool)
    filtered_data = DataLoader.filter_and_construct(dataset, metadata_map)
    DataLoader.save_List_of_Dict(filtered_data, "cache/checkpoint/filtered_data.jsonl")
    item_list = DataConstructor.get_item_list(filtered_data)
    DataLoader.save_List_of_Dict(item_list, "cache/checkpoint/item_list.jsonl")
    train_pool, test_pool = DataConstructor.split_train_test(filtered_data)
    tokenizer = create_tokenizer()
    train_queries, train_labels, test_queries, test_labels, category_to_idx, idx_to_category = DataConstructor.prepare_classification_data(train_pool, test_pool)
    train_loader, test_loader = create_data_loaders(train_queries, train_labels, test_queries, test_labels, tokenizer)

    return train_loader, train_pool, test_loader, test_pool, category_to_idx, idx_to_category


def train_QueryClassifier(categories, train_loader, test_loader):
    print("Training QueryClassifier...")
    bert = QueryClassifier(categories)
    trainer = ModelTrainer(bert, "cuda", 1e-5)
    trainer.train(train_loader, test_loader, num_epochs=3)
    trainer.save_model()


def evaluate_QueryClassifier(categories, test_loader, test_pool, idx_to_category, k):
    print("Evaluating QueryClassifier...")
    bert = QueryClassifier(categories)
    trainer = ModelTrainer(bert, "cuda", 1e-5)
    trainer.load_model()
    # trainer.evaluate_topk_accuracy(test_loader, 1)
    # trainer.evaluate_topk_accuracy(test_loader, 2)
    # trainer.evaluate_topk_accuracy(test_loader, 3)
    class_result_list = trainer.predict_topk(test_loader, test_pool, idx_to_category, k)
    print("Class result[0]:", class_result_list[0])
    DataLoader.save_List_of_Dict(class_result_list, "cache/checkpoint/classifier_results.jsonl")


def get_embeddings():
    print("Getting query and metadata lists...")
    result_list = DataLoader.load_List_of_Dict("cache/checkpoint/classifier_results.jsonl")
    item_list = DataLoader.load_List_of_Dict("cache/checkpoint/item_list.jsonl")

    query_list, metadata_list = [], []

    for item in result_list:
        query_list.append("query: " + item['query'])
    for item in item_list:
        metadata_list.append("passage: " + item['metadata'])

    print(f"Query list length: {len(query_list)}")
    print(f"Metadata list length: {len(metadata_list)}")

    print(f"Getting query embeddings using multilingual-e5-small...")
    embedding_model = EmbeddingModel()

    query_embeddings = embedding_model.encode(query_list)
    metadata_embeddings = embedding_model.encode(metadata_list)

    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Metadata embeddings shape: {metadata_embeddings.shape}")

    query_embedding_dict = DataConstructor.get_query_embedding_dict(query_list, query_embeddings)
    item_embedding_dict = DataConstructor.get_item_embedding_dict_with_category(item_list, metadata_embeddings)

    DataLoader.save_dict(query_embedding_dict, "cache/checkpoint/query_embedding_dict.json")
    DataLoader.save_dict(item_embedding_dict, "cache/checkpoint/item_embedding_dict.json")


def analyze_performance(use_category_filtering):
    print("Analyzing performance...")
    classifier_results_path = "cache/checkpoint/classifier_results.jsonl"
    query_embedding_dict_path = "cache/checkpoint/query_embedding_dict.json"
    item_embedding_dict_path = "cache/checkpoint/item_embedding_dict.json"
    classifier_results = DataLoader.load_List_of_Dict(classifier_results_path)
    query_embedding_dict = DataLoader.load_dict(query_embedding_dict_path)
    item_embedding_dict = DataLoader.load_dict(item_embedding_dict_path)
    suffix = "" if use_category_filtering else "_without_classifier"
    performance_summary = PerformanceAnalyzer.analyze_performance(
        classifier_results, query_embedding_dict, item_embedding_dict, use_category_filtering=use_category_filtering
    )
    PerformanceAnalyzer.print_results(performance_summary)
    PerformanceAnalyzer.save_results(performance_summary, f"cache/analysis/performance_results{suffix}.json")


def visualize_performance():
    detailed_results_path = "cache/analysis/detailed_results_without_classifier.jsonl"
    output_dir = "cache/figure"
    PerformanceVisualizer.generate_all_visualizations(detailed_results_path, output_dir + "/without_classifier")
    detailed_results_path = "cache/analysis/detailed_results.jsonl"
    output_dir = "cache/figure"
    PerformanceVisualizer.generate_all_visualizations(detailed_results_path, output_dir)


if __name__ == "__main__":
    main()
