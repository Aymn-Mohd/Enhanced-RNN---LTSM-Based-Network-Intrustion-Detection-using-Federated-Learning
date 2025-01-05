import flwr as fl
import tensorflow as tf
import numpy as np
from logging import INFO
import logging
import argparse
import sys
from intrusion_detection_lstm_rnn import IntrusionDetectionSystem
import time
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                           roc_curve, auc, average_precision_score,
                           precision_score, recall_score, f1_score)
import seaborn as sns

# Configure logger
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IDS Client")

class IDSClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, train_data: tuple, test_data: tuple):
        """Initialize Federated Learning Client"""
        self.client_id = client_id
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        
        # Create results directory
        self.results_dir = f"fl_results/client_{client_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "rounds": []
        }
        
        # Initialize and build model
        logger.info(f"Client {client_id}: Initializing model...")
        self.ids = IntrusionDetectionSystem(sequence_length=5, n_components=20)
        self.model = self.ids.build_model((self.x_train.shape[1], self.x_train.shape[2]))
        
        # Save model summary
        self.save_model_summary()

    def save_model_summary(self):
        """Save model architecture summary"""
        summary_file = os.path.join(self.results_dir, "model_summary.txt")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        except UnicodeEncodeError:
            # Fallback to ASCII encoding if UTF-8 fails
            with open(summary_file, 'w', encoding='ascii', errors='replace') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def save_metrics(self, metrics_dict, filename):
        """Save metrics to a JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        except UnicodeEncodeError:
            # Fallback to ASCII encoding if UTF-8 fails
            with open(filepath, 'w', encoding='ascii', errors='replace') as f:
                json.dump(metrics_dict, f, indent=4)

    def plot_metrics(self):
        """Plot training metrics"""
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history["rounds"], self.metrics_history["accuracy"], 
                label='Training', marker='o')
        plt.plot(self.metrics_history["rounds"], self.metrics_history["val_accuracy"], 
                label='Validation', marker='s')
        plt.title('Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "plots", "accuracy.png"))
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history["rounds"], self.metrics_history["loss"], 
                label='Training', marker='o')
        plt.plot(self.metrics_history["rounds"], self.metrics_history["val_loss"], 
                label='Validation', marker='s')
        plt.title('Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "plots", "loss.png"))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.results_dir, "plots", "confusion_matrix.png"))
        plt.close()
        
        # Save confusion matrix to file
        np.savetxt(os.path.join(self.results_dir, "confusion_matrix.txt"), cm, fmt='%d')

    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve"""
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
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "plots", "roc_curve.png"))
        plt.close()

    def plot_pr_curve(self, y_true, y_prob):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "plots", "pr_curve.png"))
        plt.close()

    def get_parameters(self, config):
        """Get model parameters"""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train model on local data"""
        # Get training config
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        server_round = config["server_round"]
        
        logger.info(f"Client {self.client_id}: Starting training round {server_round}")
        
        # Set model parameters
        self.model.set_weights(parameters)
        
        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
        )
        
        # Update metrics history
        self.metrics_history["rounds"].append(server_round)
        self.metrics_history["loss"].append(history.history["loss"][-1])
        self.metrics_history["accuracy"].append(history.history["accuracy"][-1])
        self.metrics_history["val_loss"].append(history.history["val_loss"][-1])
        self.metrics_history["val_accuracy"].append(history.history["val_accuracy"][-1])
        
        # Save and plot metrics
        self.save_metrics(self.metrics_history, f"round_{server_round}_metrics.json")
        self.plot_metrics()
        
        # Log results
        logger.info(f"Client {self.client_id}: Round {server_round} - "
                   f"Loss: {history.history['loss'][-1]:.4f}, "
                   f"Accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return (
            self.model.get_weights(),
            len(self.x_train),
            {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1])
            }
        )

    def evaluate(self, parameters, config):
        """Evaluate model on local test data"""
        logger.info(f"Client {self.client_id}: Evaluating model")
        
        # Set model parameters
        self.model.set_weights(parameters)
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=1,
            batch_size=32
        )
        
        # Get predictions
        y_prob = self.model.predict(self.x_test, batch_size=32, verbose=0)
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate additional metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Plot evaluation metrics
        self.plot_confusion_matrix(self.y_test, y_pred)
        self.plot_roc_curve(self.y_test, y_prob)
        self.plot_pr_curve(self.y_test, y_prob)
        
        # Save final metrics
        final_metrics = {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        self.save_metrics(final_metrics, "final_metrics.json")
        
        logger.info(f"Client {self.client_id}: Evaluation - "
                   f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                   f"F1-Score: {f1:.4f}")
        
        return float(loss), len(self.x_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

def load_partition(client_id: int, num_partitions: int):
    """Load and partition data for a client
    
    Args:
        client_id: Client identifier
        num_partitions: Total number of partitions
        
    Returns:
        tuple: Training and test data for the client
    """
    try:
        # Initialize IDS
        ids = IntrusionDetectionSystem(sequence_length=5, n_components=20)
        
        # Load and preprocess data
        logger.info(f"Client {client_id}: Loading and preprocessing data...")
        data_train = ids.read_csv('KDDTrain+.txt')
        data_test = ids.read_csv('KDDTest+.txt')
        
        # Preprocess data
        X_train, y_train = ids.preprocess_data(data_train, training=True)
        X_test, y_test = ids.preprocess_data(data_test, training=False)
        
        # Calculate partition size
        train_partition_size = len(X_train) // num_partitions
        test_partition_size = len(X_test) // num_partitions
        
        # Get partition for this client
        train_start = client_id * train_partition_size
        train_end = train_start + train_partition_size if client_id < num_partitions - 1 else len(X_train)
        
        test_start = client_id * test_partition_size
        test_end = test_start + test_partition_size if client_id < num_partitions - 1 else len(X_test)
        
        # Log partition info
        logger.info(f"Client {client_id}: Data partition - "
                   f"Training samples: {train_end - train_start}, "
                   f"Test samples: {test_end - test_start}")
        
        # Return partitioned data
        return (
            (X_train[train_start:train_end], y_train[train_start:train_end]),
            (X_test[test_start:test_end], y_test[test_start:test_end])
        )
        
    except Exception as e:
        logger.error(f"Client {client_id}: Error loading data partition - {str(e)}")
        raise

def parse_args():
    """Parse command line arguments with better error handling"""
    parser = argparse.ArgumentParser(
        description='Flower IDS Client',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--client-id',
        type=int,
        required=True,
        help='Unique identifier for this client'
    )
    parser.add_argument(
        '--num-partitions',
        type=int,
        default=3,
        help='Number of data partitions'
    )
    parser.add_argument(
        '--server-port',
        type=int,
        default=8080,
        help='Port number of the Flower server'
    )
    parser.add_argument(
        '--server-address',
        type=str,
        default='127.0.0.1',
        help='IP address of the Flower server'
    )
    
    try:
        args = parser.parse_args()
        if args.client_id < 0:
            parser.error("Client ID must be non-negative")
        if args.num_partitions < 1:
            parser.error("Number of partitions must be positive")
        if args.server_port < 1 or args.server_port > 65535:
            parser.error("Server port must be between 1 and 65535")
        return args
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        sys.exit(1)

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load data partition
        logger.info(f"Client {args.client_id}: Loading data partition...")
        train_data, test_data = load_partition(args.client_id, args.num_partitions)
        
        # Start client
        logger.info(f"Client {args.client_id}: Initializing...")
        client = IDSClient(args.client_id, train_data, test_data)
        
        # Start Flower client with retry logic
        max_retries = 5
        retry_delay = 5  # seconds
        
        for retry in range(max_retries):
            try:
                logger.info(f"Client {args.client_id}: Connecting to server at {args.server_address}:{args.server_port} (Attempt {retry + 1}/{max_retries})")
                fl.client.start_numpy_client(
                    server_address=f"{args.server_address}:{args.server_port}",
                    client=client
                )
                break  # Connection successful
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Client {args.client_id}: Connection failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise  # Last attempt failed
        
    except KeyboardInterrupt:
        logger.warning(f"Client {args.client_id}: Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Client {args.client_id}: Error - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 