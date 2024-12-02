import os
import pandas as pd
from config import Config
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
import logging
from pathlib import Path

def process_training_results(classifier, data_yaml: str, epochs_list: list) -> Dict:
    """Process training results for multiple epoch runs."""
    results = {}

    for epochs in epochs_list:
        exp_path = classifier.train(data_yaml, epochs)
        if exp_path is None:
            print(f"Skipping results for {epochs} epochs due to training failure")
            continue

        try:
            # Get metrics from the training results
            metrics = classifier.model.metrics

            # Store the results
            results[epochs] = {
                'train_loss': metrics.box_loss if hasattr(metrics, 'box_loss') else None,
                'val_loss': metrics.val_box_loss if hasattr(metrics, 'val_box_loss') else None,
                'mAP50': metrics.maps[0] if hasattr(metrics, 'maps') else None
            }
            
            print(f"\nTraining completed for {epochs} epochs")
            if hasattr(metrics, 'maps'):
                print(f"Final mAP50: {metrics.maps[0]:.4f}")
                
        except Exception as e:
            print(f"Error processing results for {epochs} epochs: {str(e)}")
    
    return results

def load_training_results(results_dir: Path):
    """Load results from the latest training session."""
    try:
        print(f"Looking for results in directory: {results_dir}")
        
        # Find the latest exp directory
        exp_folders = [f for f in os.listdir(results_dir) if f.startswith('exp_')]
        if not exp_folders:
            print("No experiment folders found!")
            raise FileNotFoundError("No training results found")
        
        print(f"Found experiment folders: {exp_folders}")
        latest_exp = max(exp_folders, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        results_path = os.path.join(results_dir, latest_exp)
        print(f"Using latest experiment folder: {latest_exp}")
        
        # Load metrics from the results.csv
        csv_path = os.path.join(results_path, 'results.csv')
        print(f"Looking for results.csv at: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"results.csv not found at {csv_path}")
            raise FileNotFoundError(f"results.csv not found at {csv_path}")
            
        if os.path.exists(csv_path):
            print("Found results.csv file")
            df = pd.read_csv(csv_path)
            print("Available columns in CSV:", df.columns.tolist())
            
            # Calculate total losses
            train_total_loss = (df['train/box_loss'] + 
                              df['train/cls_loss'] + 
                              df['train/dfl_loss'])
            
            val_total_loss = (df['val/box_loss'] + 
                            df['val/cls_loss'] + 
                            df['val/dfl_loss'])
            
            # Create results dictionary
            results = {
                30: {  # Using 30 epochs as per your training
                    'train_loss': train_total_loss.values,
                    'val_loss': val_total_loss.values,
                    'mAP50': df['metrics/mAP50(B)'].values,
                    'precision': df['metrics/precision(B)'].values,
                    'recall': df['metrics/recall(B)'].values
                }
            }
            return results
            
    except Exception as e:
        logging.error(f"Error loading results: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def plot_training_results(results: Dict, config: Config):
    """Plot training and validation metrics."""
    if not results:
        print("No results to plot")
        return
        
    try:
        # Set figure style
        plt.style.use('default')
        plt.figure(figsize=(15, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        for epochs, metrics in results.items():
            epochs_range = range(1, len(metrics['train_loss']) + 1)
            plt.plot(epochs_range, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs_range, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Total Loss', fontsize=10)
        plt.title('Training and Validation Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot mAP50
        plt.subplot(2, 2, 2)
        for epochs, metrics in results.items():
            epochs_range = range(1, len(metrics['mAP50']) + 1)
            plt.plot(epochs_range, metrics['mAP50'], 'g-', label='mAP50', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('mAP50', fontsize=10)
        plt.title('Mean Average Precision (mAP50)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Precision and Recall
        plt.subplot(2, 2, 3)
        for epochs, metrics in results.items():
            epochs_range = range(1, len(metrics['precision']) + 1)
            plt.plot(epochs_range, metrics['precision'], 'purple', label='Precision', linewidth=2)
            plt.plot(epochs_range, metrics['recall'], 'orange', label='Recall', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Value', fontsize=10)
        plt.title('Precision and Recall', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Learning Curve (mAP50 vs Loss)
        plt.subplot(2, 2, 4)
        for epochs, metrics in results.items():
            scatter = plt.scatter(metrics['train_loss'], metrics['mAP50'], 
                                alpha=0.6, c=range(len(metrics['mAP50'])), 
                                cmap='viridis', s=50)
            plt.colorbar(scatter, label='Epoch')
                
        plt.xlabel('Training Loss', fontsize=10)
        plt.ylabel('mAP50', fontsize=10)
        plt.title('Learning Curve', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Create save directory if it doesn't exist
        try:
            # Get the training_results directory from config
            training_results_dir = config.RESULTS_DIR
            
            # Create directory if it doesn't exist
            os.makedirs(training_results_dir, exist_ok=True)
            
            # Save the plot
            save_path = os.path.join(training_results_dir, 'training_plots.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"Error saving plot: {e}")
            # Still show the plot even if saving fails
            plt.show()
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()
