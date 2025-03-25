"""
Evaluator module for jailbreak detection

This module provides functionality to evaluate the performance of jailbreak detection models
by comparing their predictions with human evaluations.
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os
import time


class JailbreakEvaluator:
    """Class to evaluate jailbreak detection models against human evaluations."""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        """
        Initialize the jailbreak evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.logger = logging.getLogger(__name__)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def load_results(self, results_path: str) -> pd.DataFrame:
        """
        Load model detection results from a JSON file.
        
        Args:
            results_path: Path to the results JSON file
            
        Returns:
            DataFrame containing the results
        """
        self.logger.info(f"Loading model results from {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load the original dataset with human evaluations.
        
        Args:
            dataset_path: Path to the dataset CSV file
            
        Returns:
            DataFrame containing the dataset
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        return pd.read_csv(dataset_path)
    
    def merge_results(self, model_results: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Merge model results with the original dataset.
        
        Args:
            model_results: DataFrame containing model results
            dataset: DataFrame containing the original dataset
            
        Returns:
            DataFrame with merged results
        """
        self.logger.info("Merging model results with dataset")
        
        # Create a new column for model predictions
        merged_df = dataset.copy()
        
        # Create a dictionary to map from response to prediction
        response_to_prediction = {}
        for _, row in model_results.iterrows():
            # Truncate response to a reasonable length for matching
            key = row['response'][:500]
            response_to_prediction[key] = row['jailbroken']
        
        # Apply the mapping to get model predictions
        def get_prediction(row):
            text = row['cleaned text'][:500]
            return response_to_prediction.get(text, None)
        
        merged_df['model_prediction'] = merged_df.apply(get_prediction, axis=1)
        
        # Drop rows where we couldn't find a match
        merged_df = merged_df.dropna(subset=['model_prediction'])
        
        self.logger.info(f"Merged {len(merged_df)} rows out of {len(dataset)} in the original dataset")
        
        return merged_df
    
    def calculate_metrics(self, merged_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate evaluation metrics comparing model predictions with human evaluations.
        
        Args:
            merged_df: DataFrame with both human evaluations and model predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Calculating evaluation metrics")
        
        y_true = merged_df['human_eval'].astype(bool).values
        y_pred = merged_df['model_prediction'].astype(bool).values
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'tn': confusion_matrix(y_true, y_pred)[0, 0],
            'fp': confusion_matrix(y_true, y_pred)[0, 1],
            'fn': confusion_matrix(y_true, y_pred)[1, 0],
            'tp': confusion_matrix(y_true, y_pred)[1, 1],
        }
        
        self.logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def evaluate_model(self, model_name: str, results_path: str, dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate a model by comparing its predictions with human evaluations.
        
        Args:
            model_name: Name of the model being evaluated
            results_path: Path to the model results JSON file
            dataset_path: Path to the dataset CSV file
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Load data
        model_results = self.load_results(results_path)
        dataset = self.load_dataset(dataset_path)
        
        # Merge results
        merged_df = self.merge_results(model_results, dataset)
        
        # Calculate metrics
        metrics = self.calculate_metrics(merged_df)
        
        # Determine confidence distribution
        confidence_counts = model_results['confidence'].value_counts().to_dict()
        
        # Prepare results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'confidence_distribution': confidence_counts,
            'timestamp': time.time(),
            'dataset_size': len(dataset),
            'results_size': len(model_results),
            'merged_size': len(merged_df),
        }
        
        # Save results
        output_path = os.path.join(self.results_dir, f"{model_name.replace('/', '_')}_evaluation.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
        
        return results
    
    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            DataFrame with model comparison
        """
        self.logger.info(f"Comparing {len(evaluation_results)} models")
        
        # Extract metrics for each model
        models_data = []
        for result in evaluation_results:
            model_data = {
                'model_name': result['model_name'],
                **result['metrics']
            }
            models_data.append(model_data)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(models_data)
        
        # Save comparison to CSV
        output_path = os.path.join(self.results_dir, "model_comparison.csv")
        comparison_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Model comparison saved to {output_path}")
        
        return comparison_df
    
    def plot_comparison(self, comparison_df: pd.DataFrame, metric: str = 'f1') -> str:
        """
        Plot model comparison based on a specified metric.
        
        Args:
            comparison_df: DataFrame with model comparison
            metric: Metric to use for comparison
            
        Returns:
            Path to the saved plot
        """
        self.logger.info(f"Plotting model comparison based on {metric}")
        
        plt.figure(figsize=(12, 6))
        
        # Sort by the specified metric
        sorted_df = comparison_df.sort_values(by=metric, ascending=False)
        
        # Create bar plot
        bars = plt.bar(sorted_df['model_name'], sorted_df[metric])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')
        
        plt.title(f'Model Comparison by {metric.capitalize()}')
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.results_dir, f"model_comparison_{metric}.png")
        plt.savefig(output_path)
        
        self.logger.info(f"Comparison plot saved to {output_path}")
        
        return output_path
    
    def identify_disagreements(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify cases where the model and human evaluators disagree.
        
        Args:
            merged_df: DataFrame with both human evaluations and model predictions
            
        Returns:
            DataFrame containing only the disagreement cases
        """
        disagreements = merged_df[merged_df['human_eval'] != merged_df['model_prediction']]
        
        self.logger.info(f"Found {len(disagreements)} disagreements out of {len(merged_df)} evaluated cases")
        
        # Save disagreements to CSV
        output_path = os.path.join(self.results_dir, "disagreements.csv")
        disagreements.to_csv(output_path, index=False)
        
        self.logger.info(f"Disagreements saved to {output_path}")
        
        return disagreements 