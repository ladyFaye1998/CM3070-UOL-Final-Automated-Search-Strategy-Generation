import time
import matplotlib.pyplot as plt
from automated_search_strategy_generator import FineTunedSearchStrategyGenerator, hybrid_search_t5

def main():
    # Sample inputs for performance evaluation
    job_desc = "Software Engineer with 5+ years experience in Python and machine learning."
    candidate_resume = (
        "John Doe\n"
        "Experience: 5 years of Python development, machine learning projects, "
        "and software engineering skills.\n"
        "Skills: Python, machine learning, data analysis, software development."
    )

    # Initialize the query generator.
    # Ensure that the fine-tuned model "fine-tuned-t5" is available in your project folder.
    generator = FineTunedSearchStrategyGenerator("fine-tuned-t5")
    
    # Number of trials for performance measurement
    n_trials = 50
    query_times = []
    print("Measuring Query Generation Time:")
    for i in range(n_trials):
        start_time = time.perf_counter()
        query, _, _ = generator.generate_boolean_query(job_desc)
        end_time = time.perf_counter()
        duration = end_time - start_time
        query_times.append(duration)
        print(f"Trial {i+1}: Query generation took {duration:.4f} seconds")
    
    avg_query_time = sum(query_times) / n_trials
    print(f"\nAverage query generation time over {n_trials} trials: {avg_query_time:.4f} seconds\n")

    # Visualize query generation times (Line Graph)
    plt.figure(figsize=(10, 5))
    plt.title("Query Generation Times per Trial")
    plt.xlabel("Trial Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("query_generation_times.png")
    plt.show()
    evaluation_times = []
    # Generate a Boolean query once, then use it for candidate evaluation trials.
    query, _, _ = generator.generate_boolean_query(job_desc)
    print("Measuring Candidate Evaluation Time:")
    for i in range(n_trials):
        start_time = time.perf_counter()
        passed, score = hybrid_search_t5(candidate_resume, query, job_desc, generator.model, generator.tokenizer)
        end_time = time.perf_counter()
        duration = end_time - start_time
        evaluation_times.append(duration)
        print(f"Trial {i+1}: Candidate evaluation took {duration:.4f} seconds")
    
    avg_eval_time = sum(evaluation_times) / n_trials
    print(f"\nAverage candidate evaluation time over {n_trials} trials: {avg_eval_time:.4f} seconds\n")

    # Visualize candidate evaluation times (Line Graph)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_trials + 1), evaluation_times, marker='o', linestyle='-')
    plt.title("Candidate Evaluation Times per Trial")
    plt.xlabel("Trial Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("candidate_evaluation_times.png")
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.hist(query_times, bins=10, alpha=0.7, label="Query Generation")
    plt.hist(evaluation_times, bins=10, alpha=0.7, label="Candidate Evaluation")
    plt.title("Performance Metrics Histogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_histogram.png")
    plt.show()

if __name__ == "__main__":
    main()
