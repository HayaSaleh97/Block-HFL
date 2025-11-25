import json
import argparse

def load_metrics(file_path="fl_metrics.json"):
    """Load metrics from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return None

def display_round_summary(round_data):
    """Display summary statistics for a round"""
    round_id = round_data.get("round_id", "Unknown")
    
    print(f"\n==== Round {round_id} Summary ====")
    
    # Global model metrics
    global_metrics = round_data.get("global_model", {})
    if global_metrics:
        print("\nGlobal Model Metrics:")
        print(f"  Accuracy:  {global_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {global_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {global_metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score:  {global_metrics.get('f1_score', 'N/A'):.4f}")
        print(f"  Latency:   {global_metrics.get('latency', 'N/A'):.4f} seconds")
    
    # Local model metrics
    local_models = round_data.get("local_models", {})
    if local_models:
        print("\nLocal Model Metrics (average across clients):")
        
        # Calculate averages
        avg_accuracy = sum(m.get("accuracy", 0) for m in local_models.values()) / len(local_models)
        avg_precision = sum(m.get("precision", 0) for m in local_models.values()) / len(local_models)
        avg_recall = sum(m.get("recall", 0) for m in local_models.values()) / len(local_models)
        avg_f1 = sum(m.get("f1_score", 0) for m in local_models.values()) / len(local_models)
        avg_latency = sum(m.get("latency", 0) for m in local_models.values()) / len(local_models)
        
        print(f"  Avg Accuracy:  {avg_accuracy:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Avg Recall:    {avg_recall:.4f}")
        print(f"  Avg F1 Score:  {avg_f1:.4f}")
        print(f"  Avg Latency:   {avg_latency:.4f} seconds")
    
    # Round latency
    total_latency = round_data.get("total_latency", "N/A")
    print(f"\nTotal Round Latency: {total_latency:.4f} seconds\n")

def display_all_metrics(metrics_data):
    """Display all metrics data"""
    if not metrics_data:
        print("No metrics data to display")
        return
    
    rounds = metrics_data.get("rounds", [])
    if not rounds:
        print("No rounds data found in metrics")
        return
    
    print(f"\n===== Federated Learning Metrics Summary ({len(rounds)} rounds) =====")
    
    # Display each round
    for round_data in rounds:
        display_round_summary(round_data)
    
    # Display overall trends
    print("\n===== Overall Performance Trends =====")
    
    # Extract accuracy trends
    global_accuracies = [r.get("global_model", {}).get("accuracy", 0) for r in rounds]
    global_latencies = [r.get("global_model", {}).get("latency", 0) for r in rounds]
    
    # Print trends
    print("\nGlobal Model Accuracy Trend:")
    for i, acc in enumerate(global_accuracies):
        print(f"  Round {i+1}: {acc:.4f}")
    
    print("\nGlobal Model Latency Trend:")
    for i, lat in enumerate(global_latencies):
        print(f"  Round {i+1}: {lat:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display FL metrics from JSON file')
    parser.add_argument('--file', type=str, default='fl_metrics.json',
                        help='Path to metrics JSON file (default: fl_metrics.json)')
    
    args = parser.parse_args()
    metrics = load_metrics(args.file)
    
    if metrics:
        display_all_metrics(metrics)