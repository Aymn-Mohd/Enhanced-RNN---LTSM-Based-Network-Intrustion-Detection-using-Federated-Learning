import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FederatedLearningAnalysis:
    def __init__(self):
        # Initialize round data
        self.rounds_data = {
            'Round 1': {
                'training': {
                    'loss': 2.3289,
                    'accuracy': 86.89,
                    'precision': 87.12,
                    'recall': 98.45,
                    'f1': 92.44
                },
                'evaluation': {
                    'loss': 0.8237,
                    'accuracy': 99.44,
                    'roc_auc': 0.47,
                    'ap': 0.99
                }
            },
            'Round 2': {
                'training': {
                    'loss': 0.5218,
                    'accuracy': 99.82,
                    'precision': 99.85,
                    'recall': 99.89,
                    'f1': 99.87
                },
                'evaluation': {
                    'loss': 0.1771,
                    'accuracy': 99.44,
                    'roc_auc': 0.49,
                    'ap': 0.99
                }
            },
            'Round 3': {
                'training': {
                    'loss': 0.2810,
                    'accuracy': 99.93,
                    'precision': 99.94,
                    'recall': 100.00,
                    'f1': 99.97
                },
                'evaluation': {
                    'loss': 0.0646,
                    'accuracy': 99.44,
                    'roc_auc': 0.52,
                    'ap': 0.99
                }
            }
        }
        
        # Final confusion matrix data
        self.confusion_matrix = np.array([
            [0, 123],
            [0, 22416]
        ])

    def plot_training_metrics(self):
        """Plot training metrics across rounds"""
        rounds = list(self.rounds_data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = [self.rounds_data[round]['training'][metric] for round in rounds]
            plt.plot(rounds, values, marker='o', label=metric.capitalize())
        
        plt.title('Training Metrics Progression')
        plt.xlabel('Rounds')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss_curves(self):
        """Plot loss curves for training and evaluation"""
        rounds = list(self.rounds_data.keys())
        training_loss = [self.rounds_data[round]['training']['loss'] for round in rounds]
        eval_loss = [self.rounds_data[round]['evaluation']['loss'] for round in rounds]

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, training_loss, marker='o', label='Training Loss')
        plt.plot(rounds, eval_loss, marker='o', label='Evaluation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self):
        """Plot the final confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Final Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def create_interactive_dashboard(self):
        """Create an interactive dashboard using plotly"""
        rounds = list(self.rounds_data.keys())
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Training Metrics', 'Loss Curves',
                                         'Evaluation Metrics', 'ROC-AUC Progression'))

        # Training metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            values = [self.rounds_data[round]['training'][metric] for round in rounds]
            fig.add_trace(
                go.Scatter(x=rounds, y=values, name=f'{metric.capitalize()}'),
                row=1, col=1
            )

        # Loss curves
        training_loss = [self.rounds_data[round]['training']['loss'] for round in rounds]
        eval_loss = [self.rounds_data[round]['evaluation']['loss'] for round in rounds]
        fig.add_trace(
            go.Scatter(x=rounds, y=training_loss, name='Training Loss'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=eval_loss, name='Evaluation Loss'),
            row=1, col=2
        )

        # Evaluation metrics
        eval_acc = [self.rounds_data[round]['evaluation']['accuracy'] for round in rounds]
        eval_ap = [self.rounds_data[round]['evaluation']['ap'] for round in rounds]
        fig.add_trace(
            go.Scatter(x=rounds, y=eval_acc, name='Evaluation Accuracy'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=eval_ap, name='Average Precision'),
            row=2, col=1
        )

        # ROC-AUC progression
        roc_auc = [self.rounds_data[round]['evaluation']['roc_auc'] for round in rounds]
        fig.add_trace(
            go.Scatter(x=rounds, y=roc_auc, name='ROC-AUC'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True,
                         title_text="Federated Learning Analysis Dashboard")
        fig.show()

    def generate_report(self):
        """Generate a text report of the analysis"""
        report = []
        report.append("Federated Learning Analysis Report")
        report.append("=" * 30 + "\n")

        for round_name, data in self.rounds_data.items():
            report.append(f"{round_name} Results:")
            report.append("-" * 20)
            report.append("\nTraining Metrics:")
            for metric, value in data['training'].items():
                report.append(f"- {metric.capitalize()}: {value:.4f}")
            
            report.append("\nEvaluation Metrics:")
            for metric, value in data['evaluation'].items():
                report.append(f"- {metric.capitalize()}: {value:.4f}")
            report.append("\n")

        return "\n".join(report)

def main():
    # Create analysis object
    analysis = FederatedLearningAnalysis()
    
    # Generate visualizations
    analysis.plot_training_metrics()
    analysis.plot_loss_curves()
    analysis.plot_confusion_matrix()
    
    # Create interactive dashboard
    analysis.create_interactive_dashboard()
    
    # Generate and print report
    print(analysis.generate_report())

if __name__ == "__main__":
    main()
