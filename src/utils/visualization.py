import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np

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
                'train_loss': metrics.box_loss if hasattr(metrics, 'box_loss') else np.array([]),
                'val_loss': metrics.val_box_loss if hasattr(metrics, 'val_box_loss') else np.array([]),
                'mAP50': metrics.maps[0] if hasattr(metrics, 'maps') else np.array([])
            }
            
            print(f"\nTraining completed for {epochs} epochs")
            if hasattr(metrics, 'maps'):
                print(f"Final mAP50: {metrics.maps[0]:.4f}")
                
        except Exception as e:
            print(f"Error processing results for {epochs} epochs: {str(e)}")
    
    return results

def plot_training_results(results: Dict):
    """Plot training and validation metrics."""
    if not results:
        print("No results to plot")
        return
        
    try:
        plt.figure(figsize=(15, 5))
        
        # Create subplots for different metrics
        plt.subplot(1, 2, 1)
        for epochs, metrics in results.items():
            if len(metrics['train_loss']) > 0:
                plt.plot(metrics['train_loss'], label=f'Train Loss ({epochs} epochs)')
            if len(metrics['val_loss']) > 0:
                plt.plot(metrics['val_loss'], label=f'Val Loss ({epochs} epochs)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot mAP50
        plt.subplot(1, 2, 2)
        for epochs, metrics in results.items():
            if len(metrics['mAP50']) > 0:
                plt.plot(metrics['mAP50'], label=f'mAP50 ({epochs} epochs)')
        
        plt.xlabel('Epoch')
        plt.ylabel('mAP50')
        plt.title('Mean Average Precision (mAP50)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting results: {str(e)}")