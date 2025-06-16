# STSERF: Sequential Triple-Stage Embedding Retrieval Framework for Obscure Query Matching

E-commerce platforms increasingly leverage natural language processing (NLP) to enhance product retrieval and recommendation tasks. Traditional keyword-based systems often fail to capture the nuanced semantics of user queries, especially in complex or ambiguous contexts. To address this challenge, we utilize the Amazon-C4 test set from McAuley’s Lab, which comprises ambiguous queries to simulate real-world scenarios.

We propose a novel two-step retrieval framework that combines category prediction using a fine-tuned BERT model with similarity matching based on the E5 embedding model. This approach optimizes retrieval efficiency while maintaining high accuracy, achieving a test set accuracy of 74.07% for top-200 retrieval tasks.

Furthermore, we introduce a strategy to enrich the dataset by incorporating top-ranked items from our model’s predictions as additional ground-truth candidates. This paper provides qualitative examples and actionable insights into how dataset construction impacts model performance, laying the groundwork for enhancing future product retrieval systems.

To run the code, you need to manually download the dataset and place it in the same directory as `main.py`. You can choose to run the steps sequentially or all at once by commenting or uncommenting relevant sections in `main.py`. The system will automatically create a `cache` directory, so please ensure you have sufficient disk space.
