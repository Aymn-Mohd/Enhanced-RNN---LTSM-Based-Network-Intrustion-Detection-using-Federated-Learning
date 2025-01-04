import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings and set pandas display options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

class IntrusionDetectionSystem:
    def __init__(self, sequence_length=5, n_components=20):
        """Initialize the Intrusion Detection System
        
        Args:
            sequence_length (int): Length of sequences for LSTM
            n_components (int): Number of components for PCA
        """
        self.model = None
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.history = None
        self.sequence_length = sequence_length
        self.pca = PCA(n_components=n_components)
        self.columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        self.cat_cols = ['protocol_type', 'service', 'flag']
        self.num_cols = [col for col in self.columns if col not in self.cat_cols + ['label']]
    
    def read_csv(self, filepath):
        """Read CSV file with proper column names"""
        try:
            df = pd.read_csv(filepath, names=self.columns)
            print(f"\nDataset shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            return df
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            raise
    
    def prepare_sequences(self, X, y):
        """Prepare sequences for RNN"""
        sequences = []
        labels = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:(i + self.sequence_length)])
            labels.append(y[i + self.sequence_length - 1])
            
        return np.array(sequences), np.array(labels)
    
    def plot_pca_explained_variance(self, X):
        """Plot PCA explained variance ratio
        
        Args:
            X (array): Input features
        """
        # Fit PCA and get explained variance ratio
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), 
                cumulative_variance_ratio, 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', 
                   label='95% Explained Variance')
        plt.grid(True)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.legend()
        plt.savefig('pca_explained_variance.png')
        plt.close()
        
        print(f"\nTotal explained variance with {len(explained_variance_ratio)} components: "
              f"{cumulative_variance_ratio[-1]:.4f}")

    def preprocess_data(self, dataframe, training=True):
        """Preprocess the data and create sequences for RNN with PCA
        
        Args:
            dataframe (DataFrame): Input data
            training (bool): Whether in training mode
        
        Returns:
            tuple: Processed features and labels
        """
        try:
            df = dataframe.copy()
            
            # Handle categorical columns with LabelEncoder
            for column in df.columns:
                if df[column].dtype == type(object):
                    if training:
                        le = LabelEncoder()
                        df[column] = le.fit_transform(df[column].astype(str))
                        self.label_encoders[column] = le
                    else:
                        le = self.label_encoders[column]
                        df[column] = df[column].astype(str)
                        df[column] = df[column].map(lambda x: x if x in le.classes_ else le.classes_[0])
                        df[column] = le.transform(df[column])
            
            # Convert all columns to float
            for col in df.columns:
                df[col] = df[col].astype(float)
            
            # Scale all features except label
            features = df.columns.difference(['label'])
            if training:
                df[features] = self.scaler.fit_transform(df[features])
            else:
                df[features] = self.scaler.transform(df[features])
            
            # Get features and labels
            X = df.drop('label', axis=1).values
            y = df['label'].values
            
            # Apply PCA
            if training:
                X_pca = self.pca.fit_transform(X)
                self.plot_pca_explained_variance(X)
            else:
                X_pca = self.pca.transform(X)
            
            # Convert labels to binary (0 for normal, 1 for attack)
            y = (y != 0).astype(int)
            
            # Create sequences for RNN
            X_seq, y_seq = self.prepare_sequences(X_pca, y)
            
            print("\nPreprocessing summary:")
            print(f"Original shape: {X.shape}")
            print(f"PCA shape: {X_pca.shape}")
            print(f"Sequence shape: {X_seq.shape}")
            print(f"Labels shape: {y_seq.shape}")
            print(f"Unique labels: {np.unique(y_seq)}")
            
            return X_seq, y_seq
            
        except Exception as e:
            print(f"\nError in preprocessing:")
            print(f"DataFrame head:\n{df.head()}")
            print(f"DataFrame info:\n{df.info()}")
            print(f"Error message: {str(e)}")
            raise Exception(f"Preprocessing error: {str(e)}")
    
    def build_model(self, input_shape):
        """Build the LSTM model"""
        model = Sequential([
            # First LSTM layer
            LSTM(128, input_shape=input_shape, return_sequences=True,
                 kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True,
                 kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM layer with Bidirectional wrapper
            Bidirectional(LSTM(32, return_sequences=True,
                             kernel_regularizer=regularizers.l2(0.01))),
            BatchNormalization(),
            Dropout(0.3),
            
            # Fourth LSTM layer
            LSTM(16, return_sequences=False,
                 kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers for classification
            Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        model.summary()
        return model
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=3,
                min_lr=0.00001,
                mode='max',
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_roc_curve(self, y_true, y_pred_proba, title_suffix=''):
        """Plot ROC curve
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Predicted probabilities
            title_suffix (str): Suffix for plot title
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve {title_suffix}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'roc_curve{title_suffix.lower().replace(" ", "_")}.png')
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, title_suffix=''):
        """Plot Precision-Recall curve
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Predicted probabilities
            title_suffix (str): Suffix for plot title
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve {title_suffix}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(f'pr_curve{title_suffix.lower().replace(" ", "_")}.png')
        plt.close()

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance with extended metrics
        
        Args:
            X_train (array): Training features
            X_test (array): Test features
            y_train (array): Training labels
            y_test (array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions
        train_probs = self.model.predict(X_train)
        test_probs = self.model.predict(X_test)
        
        # Convert to binary predictions
        y_train_pred = (train_probs > 0.5).astype(int).ravel()
        y_test_pred = (test_probs > 0.5).astype(int).ravel()
        
        # Ensure labels are 1D arrays
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'test_f1': f1_score(y_test, y_test_pred)
        }
        
        # Plot ROC and PR curves
        self.plot_roc_curve(y_train, train_probs, ' (Training)')
        self.plot_roc_curve(y_test, test_probs, ' (Test)')
        self.plot_precision_recall_curve(y_train, train_probs, ' (Training)')
        self.plot_precision_recall_curve(y_test, test_probs, ' (Test)')
        
        # Plot confusion matrices
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        cm_train = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Training)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()
        
        # Print detailed metrics
        print("\n=== Model Evaluation Metrics ===")
        print("\nTraining Metrics:")
        print(f"Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"Precision: {metrics['train_precision']*100:.2f}%")
        print(f"Recall: {metrics['train_recall']*100:.2f}%")
        print(f"F1-Score: {metrics['train_f1']*100:.2f}%")
        
        print("\nTest Metrics:")
        print(f"Accuracy: {metrics['test_accuracy']*100:.2f}%")
        print(f"Precision: {metrics['test_precision']*100:.2f}%")
        print(f"Recall: {metrics['test_recall']*100:.2f}%")
        print(f"F1-Score: {metrics['test_f1']*100:.2f}%")
        
        # Save metrics to file
        with open('evaluation_metrics.txt', 'w') as f:
            f.write("=== Model Evaluation Metrics ===\n\n")
            f.write("Training Metrics:\n")
            f.write(f"Accuracy: {metrics['train_accuracy']*100:.2f}%\n")
            f.write(f"Precision: {metrics['train_precision']*100:.2f}%\n")
            f.write(f"Recall: {metrics['train_recall']*100:.2f}%\n")
            f.write(f"F1-Score: {metrics['train_f1']*100:.2f}%\n\n")
            f.write("Test Metrics:\n")
            f.write(f"Accuracy: {metrics['test_accuracy']*100:.2f}%\n")
            f.write(f"Precision: {metrics['test_precision']*100:.2f}%\n")
            f.write(f"Recall: {metrics['test_recall']*100:.2f}%\n")
            f.write(f"F1-Score: {metrics['test_f1']*100:.2f}%\n")
        
        return metrics

def main():
    try:
        import time
        start_time = time.time()
        
        print("=== Starting Intrusion Detection System ===")
        
        # Initialize IDS with sequence length
        sequence_length = 5
        n_components = 20
        ids = IntrusionDetectionSystem(sequence_length=sequence_length, n_components=n_components)
        
        # Load and preprocess training data
        print("\nLoading and preprocessing training data...")
        data_train = ids.read_csv('KDDTrain+.txt')
        preprocessing_start = time.time()
        X_train, y_train = ids.preprocess_data(data_train, training=True)
        preprocessing_time = time.time() - preprocessing_start
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Build and train model
        print("\nBuilding and training model...")
        training_start = time.time()
        ids.build_model((X_train.shape[1], X_train.shape[2]))  # Shape: (sequence_length, features)
        history = ids.train(X_train, y_train)
        training_time = time.time() - training_start
        
        # Load and preprocess test data
        print("\nLoading and preprocessing test data...")
        data_test = ids.read_csv('KDDTest+.txt')
        test_preprocessing_start = time.time()
        X_test, y_test = ids.preprocess_data(data_test, training=False)
        test_preprocessing_time = time.time() - test_preprocessing_start
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluation_start = time.time()
        metrics = ids.evaluate_model(X_train, X_test, y_train, y_test)
        evaluation_time = time.time() - evaluation_start
        
        total_time = time.time() - start_time
        
        # Print final summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        
        print("\nModel Parameters:")
        print(f"Sequence Length: {sequence_length}")
        print(f"PCA Components: {n_components}")
        print(f"Batch Size: 32")
        print(f"Initial Learning Rate: 0.001")
        print(f"Dropout Rate: 0.3")
        
        print("\nData Dimensions:")
        print(f"Original Training Data Shape: {data_train.shape}")
        print(f"Original Test Data Shape: {data_test.shape}")
        print(f"After PCA - Training Shape: {X_train.shape}")
        print(f"After PCA - Test Shape: {X_test.shape}")
        
        print("\nProcessing Times:")
        print(f"Training Data Preprocessing: {preprocessing_time:.2f} seconds")
        print(f"Model Training: {training_time:.2f} seconds")
        print(f"Test Data Preprocessing: {test_preprocessing_time:.2f} seconds")
        print(f"Model Evaluation: {evaluation_time:.2f} seconds")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        
        print("\nConfusion Matrix Analysis:")
        cm_train = confusion_matrix(y_train.ravel(), (ids.model.predict(X_train) > 0.5).astype(int).ravel())
        cm_test = confusion_matrix(y_test.ravel(), (ids.model.predict(X_test) > 0.5).astype(int).ravel())
        
        print("\nTraining Confusion Matrix:")
        print("True Negatives:", cm_train[0,0])
        print("False Positives:", cm_train[0,1])
        print("False Negatives:", cm_train[1,0])
        print("True Positives:", cm_train[1,1])
        
        print("\nTest Confusion Matrix:")
        print("True Negatives:", cm_test[0,0])
        print("False Positives:", cm_test[0,1])
        print("False Negatives:", cm_test[1,0])
        print("True Positives:", cm_test[1,1])
        
        # Calculate additional metrics
        train_accuracy = (cm_train[0,0] + cm_train[1,1]) / np.sum(cm_train)
        test_accuracy = (cm_test[0,0] + cm_test[1,1]) / np.sum(cm_test)
        
        train_precision = cm_train[1,1] / (cm_train[1,1] + cm_train[0,1])
        test_precision = cm_test[1,1] / (cm_test[1,1] + cm_test[0,1])
        
        train_recall = cm_train[1,1] / (cm_train[1,1] + cm_train[1,0])
        test_recall = cm_test[1,1] / (cm_test[1,1] + cm_test[1,0])
        
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print("\nDetailed Metrics:")
        print(f"Training - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, "
              f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
              f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        # Save all metrics to file
        with open('detailed_results.txt', 'w') as f:
            f.write("="*50 + "\n")
            f.write("INTRUSION DETECTION SYSTEM - DETAILED RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("Model Parameters:\n")
            f.write(f"Sequence Length: {sequence_length}\n")
            f.write(f"PCA Components: {n_components}\n")
            f.write(f"Batch Size: 32\n")
            f.write(f"Initial Learning Rate: 0.001\n")
            f.write(f"Dropout Rate: 0.3\n\n")
            
            f.write("Data Dimensions:\n")
            f.write(f"Original Training Data Shape: {data_train.shape}\n")
            f.write(f"Original Test Data Shape: {data_test.shape}\n")
            f.write(f"After PCA - Training Shape: {X_train.shape}\n")
            f.write(f"After PCA - Test Shape: {X_test.shape}\n\n")
            
            f.write("Processing Times:\n")
            f.write(f"Training Data Preprocessing: {preprocessing_time:.2f} seconds\n")
            f.write(f"Model Training: {training_time:.2f} seconds\n")
            f.write(f"Test Data Preprocessing: {test_preprocessing_time:.2f} seconds\n")
            f.write(f"Model Evaluation: {evaluation_time:.2f} seconds\n")
            f.write(f"Total Execution Time: {total_time:.2f} seconds\n\n")
            
            f.write("Training Confusion Matrix:\n")
            f.write(f"True Negatives: {cm_train[0,0]}\n")
            f.write(f"False Positives: {cm_train[0,1]}\n")
            f.write(f"False Negatives: {cm_train[1,0]}\n")
            f.write(f"True Positives: {cm_train[1,1]}\n\n")
            
            f.write("Test Confusion Matrix:\n")
            f.write(f"True Negatives: {cm_test[0,0]}\n")
            f.write(f"False Positives: {cm_test[0,1]}\n")
            f.write(f"False Negatives: {cm_test[1,0]}\n")
            f.write(f"True Positives: {cm_test[1,1]}\n\n")
            
            f.write("Detailed Metrics:\n")
            f.write(f"Training - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, "
                   f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}\n")
            f.write(f"Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
                   f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
