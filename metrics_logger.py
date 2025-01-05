import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                           roc_curve, auc, average_precision_score)

class MetricsLogger:
    """Utility class for logging and visualizing federated learning metrics"""
    
    def __init__(self, base_dir="fl_results", client_id=None):
        """Initialize the metrics logger
        
        Args:
            base_dir (str): Base directory for saving results
            client_id (int, optional): Client ID if this is a client logger
        """
        self.base_dir = base_dir
        self.client_id = client_id
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "rounds": [],
            "predictions": [],
            "true_labels": [],
            "probabilities": []
        }
        
        # Create directory structure
        if client_id is not None:
            self.results_dir = os.path.join(base_dir, f"client_{client_id}")
        else:
            self.results_dir = os.path.join(base_dir, "server")
            
        self.plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.results_dir, "metrics.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Get logger
        if self.client_id is not None:
            self.logger = logging.getLogger(f"FL_Client_{self.client_id}")
        else:
            self.logger = logging.getLogger("FL_Server")
        
        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
    
    def log_metrics(self, round_number, metrics_dict):
        """Log metrics for a training round
        
        Args:
            round_number (int): Current round number
            metrics_dict (dict): Dictionary containing metrics
        """
        self.metrics["rounds"].append(round_number)
        
        for metric_name, metric_value in metrics_dict.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].append(metric_value)
        
        # Log to file
        self.logger.info(f"Round {round_number} metrics: {metrics_dict}")
        
        # Save metrics to JSON
        self.save_metrics()
        
        # Update plots
        self.plot_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_metrics(self):
        """Create and save visualization plots"""
        # Set style
        plt.style.use('seaborn')
        
        # Create accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["rounds"], self.metrics["accuracy"], 
                label='Training Accuracy', marker='o')
        if self.metrics["val_accuracy"]:
            plt.plot(self.metrics["rounds"], self.metrics["val_accuracy"], 
                    label='Validation Accuracy', marker='s')
        plt.title('Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'accuracy.png'))
        plt.close()
        
        # Create loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["rounds"], self.metrics["loss"], 
                label='Training Loss', marker='o')
        if self.metrics["val_loss"]:
            plt.plot(self.metrics["rounds"], self.metrics["val_loss"], 
                    label='Validation Loss', marker='s')
        plt.title('Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'loss.png'))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, title_prefix=""):
        """Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title_prefix: Prefix for the plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{title_prefix}Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.plots_dir, f'{title_prefix.lower()}confusion_matrix.png'))
        plt.close()
        
        # Save confusion matrix to file
        cm_file = os.path.join(self.results_dir, f"{title_prefix.lower()}confusion_matrix.txt")
        with open(cm_file, 'w') as f:
            f.write(f"Confusion Matrix:\n{cm}")
    
    def plot_roc_curve(self, y_true, y_prob, title_prefix=""):
        """Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title_prefix: Prefix for the plot title
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title_prefix}Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'{title_prefix.lower()}roc_curve.png'))
        plt.close()
        
        # Save ROC data to file
        roc_file = os.path.join(self.results_dir, f"{title_prefix.lower()}roc_data.txt")
        with open(roc_file, 'w') as f:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write("FPR,TPR\n")
            for x, y in zip(fpr, tpr):
                f.write(f"{x:.4f},{y:.4f}\n")
    
    def plot_pr_curve(self, y_true, y_prob, title_prefix=""):
        """Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title_prefix: Prefix for the plot title
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title_prefix}Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'{title_prefix.lower()}pr_curve.png'))
        plt.close()
        
        # Save PR data to file
        pr_file = os.path.join(self.results_dir, f"{title_prefix.lower()}pr_data.txt")
        with open(pr_file, 'w') as f:
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write("Precision,Recall\n")
            for p, r in zip(precision, recall):
                f.write(f"{p:.4f},{r:.4f}\n")
    
    def plot_pca_components(self, pca, X, n_components=None):
        """Plot PCA explained variance and cumulative variance
        
        Args:
            pca: Fitted PCA object
            X: Input data
            n_components: Number of components to plot
        """
        if n_components is None:
            n_components = len(pca.explained_variance_ratio_)
        
        # Plot explained variance ratio
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, n_components + 1), 
                pca.explained_variance_ratio_[:n_components])
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')
        
        # Plot cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_components + 1), 
                np.cumsum(pca.explained_variance_ratio_[:n_components]),
                'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'pca_analysis.png'))
        plt.close()
        
        # Save PCA data to file
        pca_file = os.path.join(self.results_dir, "pca_data.txt")
        with open(pca_file, 'w') as f:
            f.write("PCA Analysis Results:\n")
            f.write("\nExplained Variance Ratio:\n")
            for i, var in enumerate(pca.explained_variance_ratio_[:n_components], 1):
                f.write(f"PC{i}: {var:.4f}\n")
            f.write("\nCumulative Explained Variance:\n")
            cum_var = np.cumsum(pca.explained_variance_ratio_[:n_components])
            for i, var in enumerate(cum_var, 1):
                f.write(f"Components 1-{i}: {var:.4f}\n")
    
    def log_predictions(self, y_true, y_pred, y_prob):
        """Log prediction results for later analysis
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        """
        self.metrics["true_labels"] = y_true.tolist()
        self.metrics["predictions"] = y_pred.tolist()
        self.metrics["probabilities"] = y_prob.tolist()
        
        # Plot evaluation metrics
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_prob)
        self.plot_pr_curve(y_true, y_prob)
    
    def log_final_metrics(self, final_metrics):
        """Log final evaluation metrics
        
        Args:
            final_metrics (dict): Dictionary containing final metrics
        """
        # Save final metrics
        final_metrics_file = os.path.join(self.results_dir, "final_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        # Log to file
        self.logger.info(f"Final metrics: {final_metrics}")
        
        # If predictions are available, plot evaluation curves
        if "true_labels" in final_metrics and "predictions" in final_metrics:
            self.plot_confusion_matrix(
                final_metrics["true_labels"],
                final_metrics["predictions"],
                "Final_"
            )
            if "probabilities" in final_metrics:
                self.plot_roc_curve(
                    final_metrics["true_labels"],
                    final_metrics["probabilities"],
                    "Final_"
                )
                self.plot_pr_curve(
                    final_metrics["true_labels"],
                    final_metrics["probabilities"],
                    "Final_"
                )
    
    def log_model_summary(self, model):
        """Log model architecture summary
        
        Args:
            model: Keras model
        """
        summary_file = os.path.join(self.results_dir, "model_summary.txt")
        
        # Redirect model.summary() to file
        from contextlib import redirect_stdout
        with open(summary_file, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        # Log to file
        self.logger.info(f"Model summary saved to {summary_file}") 