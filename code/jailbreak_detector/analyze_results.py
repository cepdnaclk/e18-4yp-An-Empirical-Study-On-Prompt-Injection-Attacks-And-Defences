#!/usr/bin/env python3
"""
Analyze Jailbreak Detection Results and Generate Visualizations

This script analyzes the merged results from the jailbreak detection pipeline
and generates various visualizations and statistics.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from typing import Dict, List, Any, Tuple

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze jailbreak detection results")
    parser.add_argument("--results-file", "-r", 
                        default="analysis_results/opt-1.3b_merged_results.json",
                        help="Path to the merged results file")
    parser.add_argument("--output-dir", "-o",
                        default="analysis_visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--model-name", "-m",
                        default=None,
                        help="Override model name for plots (defaults to name from results)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for output images")
    parser.add_argument("--format", default="png",
                        help="Output image format (png, jpg, pdf, svg)")
    return parser.parse_args()

def load_results(results_file: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load the results file and convert to DataFrame."""
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results_data["results"])
    
    # Fill any missing data if present
    results_df.fillna({'human_eval': 0, 'model_eval': 0, 'agreement': False}, inplace=True)
    
    # Ensure correct types
    results_df['human_eval'] = results_df['human_eval'].astype(int)
    results_df['model_eval'] = results_df['model_eval'].astype(int)
    results_df['agreement'] = results_df['agreement'].astype(bool)
    
    return results_data, results_df

def generate_confusion_matrix(
    results_df: pd.DataFrame, 
    output_dir: str, 
    model_name: str,
    dpi: int = 300,
    format: str = "png"
) -> Dict[str, float]:
    """Generate and save confusion matrix visualization."""
    # Get true labels and predicted labels
    y_true = results_df['human_eval']
    y_pred = results_df['model_eval']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    # Create a prettier confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Plot a normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Seaborn heatmap
    ax = sns.heatmap(
        cm_norm, 
        annot=cm,  # Show raw counts
        fmt='d',   # Format as integers
        cmap='Blues',
        vmin=0, vmax=1,
        cbar=True,
        annot_kws={"size": 16}
    )
    
    # Labels
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    
    # Set tick labels
    tick_labels = ['Not Jailbroken (0)', 'Jailbroken (1)']
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_yticklabels(tick_labels, fontsize=12)
    
    # Add metrics text
    plt.figtext(0.5, 0.01, 
                f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}",
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_confusion_matrix.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {output_path}")
    
    return metrics

def generate_accuracy_plot(
    results_df: pd.DataFrame, 
    output_dir: str, 
    model_name: str,
    dpi: int = 300,
    format: str = "png"
):
    """Generate cumulative accuracy plot over the dataset."""
    # Sort by index to ensure ordering
    df_sorted = results_df.sort_values('index')
    
    # Calculate cumulative accuracy
    cumulative_correct = df_sorted['agreement'].cumsum()
    cumulative_total = range(1, len(df_sorted) + 1)
    cumulative_accuracy = [correct / total for correct, total in zip(cumulative_correct, cumulative_total)]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_total, cumulative_accuracy, linewidth=2)
    plt.axhline(y=cumulative_accuracy[-1], color='r', linestyle='--', alpha=0.7, 
                label=f'Final Accuracy: {cumulative_accuracy[-1]:.4f}')
    
    # Add labels and title
    plt.xlabel('Number of Samples Processed', fontsize=14)
    plt.ylabel('Cumulative Accuracy', fontsize=14)
    plt.title(f'Accuracy Progression - {model_name}', fontsize=16)
    
    # Set axis limits
    plt.ylim([0, 1.05])
    plt.xlim([0, len(df_sorted)])
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_accuracy_progression.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy progression plot to {output_path}")

def generate_roc_curve(
    results_df: pd.DataFrame, 
    output_dir: str, 
    model_name: str,
    dpi: int = 300,
    format: str = "png"
):
    """Generate ROC curve for jailbreak detection."""
    # Get true labels and predicted labels
    y_true = results_df['human_eval']
    y_pred = results_df['model_eval']
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
    
    # Add labels and title
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=12)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_roc_curve.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curve to {output_path}")

def generate_class_distribution(
    results_df: pd.DataFrame, 
    output_dir: str, 
    model_name: str,
    dpi: int = 300,
    format: str = "png"
):
    """Generate class distribution visualization."""
    # Calculate class distributions
    human_dist = results_df['human_eval'].value_counts().sort_index()
    model_dist = results_df['model_eval'].value_counts().sort_index()
    
    # Ensure both have indices 0 and 1
    for idx in [0, 1]:
        if idx not in human_dist.index:
            human_dist[idx] = 0
        if idx not in model_dist.index:
            model_dist[idx] = 0
    
    human_dist = human_dist.sort_index()
    model_dist = model_dist.sort_index()
    
    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)  # [0, 1]
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, human_dist.values, width, label='Human Evaluation', alpha=0.7)
    ax.bar(x + width/2, model_dist.values, width, label='Model Prediction', alpha=0.7)
    
    # Add counts as text on each bar
    for i, v in enumerate(human_dist.values):
        ax.text(i - width/2, v + 0.5, str(v), horizontalalignment='center', fontsize=12)
        
    for i, v in enumerate(model_dist.values):
        ax.text(i + width/2, v + 0.5, str(v), horizontalalignment='center', fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('Class Label', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Class Distribution - {model_name}', fontsize=16)
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(['Not Jailbroken (0)', 'Jailbroken (1)'], fontsize=12)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_class_distribution.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved class distribution plot to {output_path}")

def generate_agreement_bars(
    results_df: pd.DataFrame, 
    output_dir: str, 
    model_name: str,
    dpi: int = 300,
    format: str = "png"
):
    """Generate agreement bar chart by class."""
    # Group by human_eval and calculate agreement percentage
    agreement_by_class = results_df.groupby('human_eval')['agreement'].mean() * 100
    
    # Ensure both classes are present
    for idx in [0, 1]:
        if idx not in agreement_by_class.index:
            agreement_by_class[idx] = 0
    
    agreement_by_class = agreement_by_class.sort_index()
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        ['Not Jailbroken (0)', 'Jailbroken (1)'], 
        agreement_by_class.values, 
        color=['green', 'blue'], 
        alpha=0.7
    )
    
    # Add percentage text on each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            fontsize=12
        )
    
    # Add labels and title
    ax.set_xlabel('True Class', fontsize=14)
    ax.set_ylabel('Agreement Percentage (%)', fontsize=14)
    ax.set_title(f'Model-Human Agreement by Class - {model_name}', fontsize=16)
    
    # Set y-axis limits
    ax.set_ylim([0, 105])
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall accuracy text
    overall_accuracy = results_df['agreement'].mean() * 100
    plt.figtext(
        0.5, 0.01,
        f"Overall Agreement: {overall_accuracy:.1f}%",
        ha="center",
        fontsize=12,
        bbox={"facecolor":"white", "alpha":0.8, "pad":5}
    )
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_agreement_by_class.{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved agreement by class plot to {output_path}")

def save_summary_statistics(
    metrics: Dict[str, float],
    results_data: Dict[str, Any],
    results_df: pd.DataFrame,
    output_dir: str,
    model_name: str
):
    """Save summary statistics to a JSON file."""
    # Create a summary dictionary
    summary = {
        "model": model_name,
        "total_samples": len(results_df),
        "metrics": metrics,
        "class_distribution": {
            "human_evaluation": results_df['human_eval'].value_counts().to_dict(),
            "model_prediction": results_df['model_eval'].value_counts().to_dict()
        },
        "agreement_by_class": results_df.groupby('human_eval')['agreement'].mean().to_dict(),
        "dataset_info": {
            "original_dataset": results_data.get("dataset", "unknown"),
            "processing_time": results_data.get("processing_time_seconds", 0),
            "merged_from": results_data.get("merged_from", 1),
            "analysis_date": results_data.get("datetime", "unknown")
        }
    }
    
    # Add classification report
    try:
        y_true = results_df['human_eval']
        y_pred = results_df['model_eval']
        report = classification_report(y_true, y_pred, output_dict=True)
        summary["classification_report"] = report
    except Exception as e:
        print(f"Warning: Could not generate classification report: {str(e)}")
    
    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_summary_statistics.json")
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary statistics to {output_path}")
    
    # Also save a readable text version
    text_path = os.path.join(output_dir, f"{model_name.replace('/', '-')}_summary.txt")
    
    with open(text_path, 'w') as f:
        f.write(f"JAILBREAK DETECTION SUMMARY - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(results_df)}\n")
        f.write(f"Analysis date: {results_data.get('datetime', 'unknown')}\n\n")
        
        f.write("METRICS\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.replace('_', ' ').title():15s}: {value:.4f}\n")
        f.write("\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 20 + "\n")
        f.write(f"True Positives:  {metrics['true_positives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"True Negatives:  {metrics['true_negatives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n\n")
        
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 20 + "\n")
        f.write("Human Evaluation:\n")
        for label, count in sorted(summary["class_distribution"]["human_evaluation"].items()):
            f.write(f"  Class {label}: {count} samples\n")
        f.write("Model Prediction:\n")
        for label, count in sorted(summary["class_distribution"]["model_prediction"].items()):
            f.write(f"  Class {label}: {count} samples\n\n")
        
        f.write("AGREEMENT BY CLASS\n")
        f.write("-" * 20 + "\n")
        for label, agreement in sorted(summary["agreement_by_class"].items()):
            f.write(f"  Class {label}: {agreement*100:.2f}%\n")
        f.write("\n")
        
        f.write("DATASET INFO\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original dataset: {summary['dataset_info']['original_dataset']}\n")
        f.write(f"Processing time: {summary['dataset_info']['processing_time']:.2f} seconds\n")
        f.write(f"Merged from: {summary['dataset_info']['merged_from']} shards\n")
    
    print(f"Saved summary text to {text_path}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load results
    print(f"Loading results from {args.results_file}")
    results_data, results_df = load_results(args.results_file)
    
    # Extract or use provided model name
    model_name = args.model_name or results_data.get("model", "unknown_model")
    model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    
    print(f"Analyzing results for model: {model_name}")
    print(f"Total samples: {len(results_df)}")
    
    # Generate visualizations
    metrics = generate_confusion_matrix(results_df, args.output_dir, model_name, args.dpi, args.format)
    generate_accuracy_plot(results_df, args.output_dir, model_name, args.dpi, args.format)
    generate_roc_curve(results_df, args.output_dir, model_name, args.dpi, args.format)
    generate_class_distribution(results_df, args.output_dir, model_name, args.dpi, args.format)
    generate_agreement_bars(results_df, args.output_dir, model_name, args.dpi, args.format)
    
    # Save summary statistics
    save_summary_statistics(metrics, results_data, results_df, args.output_dir, model_name)
    
    print("\nAnalysis complete. All visualizations and statistics have been saved.")

if __name__ == "__main__":
    main() 