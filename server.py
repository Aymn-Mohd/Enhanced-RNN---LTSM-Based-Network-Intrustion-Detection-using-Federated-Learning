import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np
import tensorflow as tf
from logging import INFO
import logging
import socket
import argparse
import time
import sys
import os
import json
from flwr.server import Server, ServerConfig
from flwr.server.app import start_server
import matplotlib.pyplot as plt

# Configure logger
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IDS Server")

# Create results directory
RESULTS_DIR = "fl_results/server"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_metrics(metrics_dict, filename):
    """Save metrics to a JSON file"""
    filepath = os.path.join(RESULTS_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    except UnicodeEncodeError:
        # Fallback to ASCII encoding if UTF-8 fails
        with open(filepath, 'w', encoding='ascii', errors='replace') as f:
            json.dump(metrics_dict, f, indent=4)

def plot_metrics(metrics_history, metric_name, title):
    """Plot metrics over rounds"""
    plt.figure(figsize=(10, 6))
    rounds = list(range(1, len(metrics_history) + 1))
    plt.plot(rounds, metrics_history, marker='o')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f'{metric_name.lower()}_history.png'))
    plt.close()

def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available("0.0.0.0", port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

def get_on_fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "local_epochs": 1,
        "server_round": server_round,
    }
    logger.info(f"Round {server_round} configuration: batch_size={config['batch_size']}, "
                f"local_epochs={config['local_epochs']}")
    return config

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation metrics weighted by number of examples."""
    if not metrics:
        return {}
    
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    weighted_accuracy = sum(accuracies) / sum(examples)
    logger.info(f"Weighted average accuracy: {weighted_accuracy:.4f}")
    
    return {
        "accuracy": weighted_accuracy,
    }

class IDSStrategy(fl.server.strategy.FedAvg):
    """Custom Federated Learning strategy for IDS"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate model weights using weighted average."""
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            if failures:
                logger.warning(f"Failures occurred during fit:")
                for failure in failures:
                    logger.warning(f"  - Client failed with: {str(failure)}")
            return None, {}
        
        # Call aggregate_fit from parent class
        try:
            aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_parameters is not None:
                # Calculate metrics
                total_examples = sum(fit_res.num_examples for _, fit_res in results)
                weighted_metrics = {}
                
                # Calculate weighted metrics
                for _, fit_res in results:
                    weight = fit_res.num_examples / total_examples
                    for metric_name, metric_value in fit_res.metrics.items():
                        if metric_name not in weighted_metrics:
                            weighted_metrics[metric_name] = 0.0
                        weighted_metrics[metric_name] += weight * metric_value
                
                # Update metrics history
                for metric_name, value in weighted_metrics.items():
                    if metric_name in self.metrics_history:
                        self.metrics_history[metric_name].append(value)
                
                # Save current round metrics
                save_metrics(weighted_metrics, f"round_{server_round}_metrics.json")
                
                # Plot updated metrics
                for metric_name in self.metrics_history:
                    if self.metrics_history[metric_name]:
                        plot_metrics(
                            self.metrics_history[metric_name],
                            metric_name,
                            f"{metric_name} Over Rounds"
                        )
                
                # Log the results
                logger.info(f"Round {server_round} completed:")
                logger.info(f"- Aggregated weights from {len(results)} clients")
                for metric_name, metric_value in weighted_metrics.items():
                    logger.info(f"- Average {metric_name}: {metric_value:.4f}")
                
                if failures:
                    logger.warning(f"- {len(failures)} clients failed")
                    for failure in failures:
                        logger.warning(f"  - Client failed with: {str(failure)}")
            else:
                logger.warning(f"Round {server_round}: Failed to aggregate weights")
                weighted_metrics = {}
            
            return aggregated_parameters, weighted_metrics
            
        except Exception as e:
            logger.error(f"Error in aggregate_fit: {str(e)}")
            return None, {}
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation metrics."""
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results to aggregate")
            return float('inf'), {}
        
        # Calculate weighted average of metrics
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        weighted_metrics = {}
        weighted_loss = 0.0
        
        for _, eval_res in results:
            weight = eval_res.num_examples / total_examples
            weighted_loss += weight * eval_res.loss
            
            # Aggregate metrics
            for metric_name, metric_value in eval_res.metrics.items():
                if metric_name not in weighted_metrics:
                    weighted_metrics[metric_name] = 0.0
                weighted_metrics[metric_name] += weight * metric_value
        
        # Add loss to metrics
        weighted_metrics["loss"] = weighted_loss
        
        # Save evaluation metrics
        save_metrics(weighted_metrics, f"round_{server_round}_eval_metrics.json")
        
        # Log evaluation results
        logger.info(f"Round {server_round} evaluation:")
        logger.info(f"- Evaluated {len(results)} clients")
        logger.info(f"- Average loss: {weighted_loss:.4f}")
        for metric_name, metric_value in weighted_metrics.items():
            logger.info(f"- Average {metric_name}: {metric_value:.4f}")
        
        if failures:
            logger.warning(f"- {len(failures)} clients failed")
            for failure in failures:
                logger.warning(f"  - Client failed with: {str(failure)}")
        
        return weighted_loss, weighted_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flower IDS Server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum number of clients')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds')
    args = parser.parse_args()

    try:
        # Find available port
        port = find_available_port(args.port)
        if port != args.port:
            logger.warning(f"Port {args.port} is in use. Using port {port} instead.")
        
        # Define strategy
        strategy = IDSStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args.min_clients,
            min_evaluate_clients=args.min_clients,
            min_available_clients=args.min_clients,
            on_fit_config_fn=get_on_fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=None,  # Let clients initialize their own models
        )

        # Start server
        logger.info("="*50)
        logger.info("Starting Federated Learning Server")
        logger.info("="*50)
        logger.info(f"Port: {port}")
        logger.info(f"Minimum clients: {args.min_clients}")
        logger.info(f"Number of rounds: {args.rounds}")
        logger.info("="*50)
        
        # Configure server
        server_config = ServerConfig(num_rounds=args.rounds)
        
        # Start Flower server
        start_server(
            server_address=f"0.0.0.0:{port}",
            config=server_config,
            strategy=strategy,
            grpc_max_message_length=536870912  # 512MB
        )

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 