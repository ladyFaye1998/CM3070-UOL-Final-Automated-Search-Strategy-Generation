#!/usr/bin/env python3
"""
evaluation.py

This script evaluates the Automated Search Strategy Generator & Resume Analyzer system.
The evaluation comprises two major components:

1. Query Generation Quality:
   - Syntactical Accuracy: Checks that generated Boolean queries have balanced parentheses.
   - Semantic Relevance: Compares automated queries with manually generated queries using Jaccard similarity.
   - Response Time: Measures the time taken to generate queries.

2. Candidate Matching Performance:
   - Precision, Recall, and F1 Score: Compares system candidate evaluation results with ground truth labels.
   - Response Time: Measures the time taken for candidate evaluation.
   - (Optional) Parameter Tuning: Could be implemented to vary the weighting parameters (α and β).

The script assumes a benchmark dataset CSV file named "benchmark_dataset.csv" exists in the same directory.
The CSV is expected to have the following columns:
    - job_description: The input job description text.
    - manual_query: The manually generated Boolean query (ground truth). (If missing, 'query' will be used.)
    - candidate_resume: The candidate resume text.
    - label: Ground truth binary label (1 for relevant, 0 for non-relevant).

Placeholders are marked as [Placeholder] where you can insert additional information.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from automated_search_strategy_generator import FineTunedSearchStrategyGenerator, hybrid_search_t5

# ----------------------------
# Utility Functions
# ----------------------------

def is_balanced(query):
    """Check if parentheses in the query are balanced."""
    stack = []
    for ch in query:
        if ch == '(':
            stack.append(ch)
        elif ch == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0

def jaccard_similarity(str1, str2):
    """Compute Jaccard similarity between two strings based on token overlap."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

# ----------------------------
# Evaluation Functions
# ----------------------------

def evaluate_query_generation(generator, job_desc, manual_query, n_trials=50):
    """
    Evaluate the quality and response time of query generation.
    
    Returns:
        avg_time: Average generation time.
        syntax_accuracy: Percentage of queries with balanced parentheses.
        avg_similarity: Average Jaccard similarity between automated and manual queries.
        times: List of per-trial response times.
        similarities: List of per-trial Jaccard similarities.
    """
    times = []
    similarities = []
    balanced_count = 0
    
    for _ in range(n_trials):
        start = time.perf_counter()
        auto_query, _, _ = generator.generate_boolean_query(job_desc)
        end = time.perf_counter()
        duration = end - start
        times.append(duration)
        
        if is_balanced(auto_query):
            balanced_count += 1
        
        similarity = jaccard_similarity(auto_query, manual_query)
        similarities.append(similarity)
    
    avg_time = sum(times) / n_trials
    syntax_accuracy = balanced_count / n_trials * 100
    avg_similarity = sum(similarities) / n_trials
    return avg_time, syntax_accuracy, avg_similarity, times, similarities

def evaluate_candidate_matching(generator, job_desc, candidate_resume, ground_truth, n_trials=50, threshold=0.5):
    """
    Evaluate candidate matching performance by running hybrid search over n_trials.
    
    Returns:
        avg_time: Average evaluation time.
        pred_labels: List of binary predictions (1 if relevant, 0 if not) from each trial.
        times: List of per-trial evaluation times.
    """
    times = []
    pred_labels = []
    
    query, _, _ = generator.generate_boolean_query(job_desc)
    
    for _ in range(n_trials):
        start = time.perf_counter()
        passed, score = hybrid_search_t5(candidate_resume, query, job_desc, generator.model, generator.tokenizer)
        end = time.perf_counter()
        duration = end - start
        times.append(duration)
        pred = 1 if score >= threshold else 0
        pred_labels.append(pred)
    
    avg_time = sum(times) / n_trials
    return avg_time, pred_labels, times

# ----------------------------
# Main Evaluation Script
# ----------------------------

def main():
    # Load benchmark dataset (CSV file)
    dataset_path = "benchmark_dataset.csv"  # [Placeholder: Adjust dataset path as needed]
    df = pd.read_csv(dataset_path)
    
    # Initialize the query generator (ensure that the fine-tuned model "fine-tuned-t5" is available)
    generator = FineTunedSearchStrategyGenerator("fine-tuned-t5")
    
    query_times_all = []
    query_similarities_all = []
    syntax_accuracies = []
    candidate_times_all = []
    all_true_labels = []
    all_pred_labels = []
    
    n_trials = 20  # Number of trials per evaluation case
    
    for idx, row in df.iterrows():
        job_desc = row['job_description']
        # Use "manual_query" if it exists, otherwise fallback to "query"
        manual_query = row.get('manual_query', row['query'])
        candidate_resume = row['candidate_resume']
        true_label = row['label']
        
        avg_q_time, syntax_acc, avg_similarity, times, similarities = evaluate_query_generation(generator, job_desc, manual_query, n_trials)
        query_times_all.extend(times)
        query_similarities_all.extend(similarities)
        syntax_accuracies.append(syntax_acc)
        
        avg_c_time, pred_labels, c_times = evaluate_candidate_matching(generator, job_desc, candidate_resume, true_label, n_trials)
        candidate_times_all.extend(c_times)
        pred_final = max(set(pred_labels), key=pred_labels.count)
        all_pred_labels.append(pred_final)
        all_true_labels.append(true_label)
    
    precision = precision_score(all_true_labels, all_pred_labels, zero_division=0) * 100
    recall = recall_score(all_true_labels, all_pred_labels, zero_division=0) * 100
    f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0) * 100

    print("Query Generation Evaluation:")
    print(f"Average Response Time: {sum(query_times_all)/len(query_times_all):.4f} seconds")
    print(f"Average Syntactical Accuracy: {sum(syntax_accuracies)/len(syntax_accuracies):.2f}%")
    print(f"Average Semantic Relevance (Jaccard Similarity): {sum(query_similarities_all)/len(query_similarities_all):.4f}")
    
    print("\nCandidate Matching Evaluation:")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Average Evaluation Time: {sum(candidate_times_all)/len(candidate_times_all):.4f} seconds")
    
    plt.figure(figsize=(10, 5))
    plt.hist(query_times_all, bins=10, alpha=0.7, label="Query Generation Times")
    plt.hist(candidate_times_all, bins=10, alpha=0.7, label="Candidate Evaluation Times")
    plt.title("Performance Metrics Histogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FFC107'])
    plt.title("Candidate Matching Metrics")
    plt.ylabel("Percentage")
    plt.ylim([0, 100])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
