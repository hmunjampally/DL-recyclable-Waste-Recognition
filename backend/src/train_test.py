from config import Config
from data_handlers.dataset import DatasetManager
from models.classifier import GarbageClassifier
from utils.environment import setup_environment
from utils.visualization import process_training_results, plot_training_results
import logging
import argparse
from pathlib import Path
import yaml
import os
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(config: Config):
    """Training function."""
    try:
        dataset_manager = DatasetManager(config)
        
        logging.info("Validating dataset structure...")
        dataset_manager.validate_dataset(config.ORIGINAL_DATASET)
        
        logging.info("Creating dataset YAML...")
        data_yaml = dataset_manager._create_actual_dataset_yaml(config.ORIGINAL_DATASET)

        logging.info("Initializing GarbageClassifier...")
        classifier = GarbageClassifier(config)
        
        logging.info("Processing training results...")
        results = process_training_results(
            classifier,
            str(data_yaml),
            epochs_list=[config.DEFAULT_EPOCHS])

        logging.info("Plot results...")
        plot_training_results(results, config)
        
        logging.info("Training completed successfully!")
        
        return classifier

    except Exception as e:
        logging.error("Error in training execution", exc_info=True)
        raise

def test_model(config: Config, image_path: str, conf_threshold: float = 0.25):
    """Testing function."""
    try:
        logging.info("Initializing GarbageClassifier for testing...")
        classifier = GarbageClassifier(config)
        
        logging.info(f"Testing image: {image_path}")
        classifier.test_single_image(image_path, conf_threshold)
        
    except Exception as e:
        logging.error("Error in testing execution", exc_info=True)
        raise

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

def plot_only(config: Config):
    """Function to only plot previous training results."""
    try:
        # Debug information
        print("\nDebugging Information:")
        print("Current working directory:", os.getcwd())
        print("\nContent of training_results directory:")
        training_results_dir = os.path.join(os.getcwd(), 'training_results')
        if os.path.exists(training_results_dir):
            print(os.listdir(training_results_dir))
            print("\nFull path to training_results:", os.path.abspath(training_results_dir))
            
            # Check content of exp directories if they exist
            for item in os.listdir(training_results_dir):
                if item.startswith('exp_'):
                    exp_path = os.path.join(training_results_dir, item)
                    print(f"\nContent of {item}:")
                    print(os.listdir(exp_path))
        else:
            print("training_results directory not found")
            print("Checking config RESULTS_DIR:", config.RESULTS_DIR)
            if os.path.exists(config.RESULTS_DIR):
                print("Content of config RESULTS_DIR:", os.listdir(config.RESULTS_DIR))
            else:
                print("config RESULTS_DIR does not exist")
            
        # Original code
        logging.info("Loading previous training results...")
        results = load_training_results(config.RESULTS_DIR)
        
        if results:
            logging.info("Plotting results...")
            plot_training_results(results, config)
        else:
            logging.error("No results found to plot")
            
    except Exception as e:
        logging.error("Error in plotting", exc_info=True)
        raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train, test, or plot results')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'plot'], required=True,
                      help='Mode to run: train, test, or plot')
    parser.add_argument('--image', type=str, help='Path to test image (required for test mode)')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold for testing (default: 0.25)')
    
    args = parser.parse_args()
    
    try:
        logging.info("Setting up environment...")
        setup_environment()

        logging.info("Initializing configuration...")
        config = Config()
        
        if args.mode == 'train':
            train_model(config)
        elif args.mode == 'test':
            if not args.image:
                raise ValueError("Test mode requires --image argument")
            test_model(config, args.image, args.conf)
        elif args.mode == 'plot':
            plot_only(config)

    except Exception as e:
        logging.error("Error in main execution", exc_info=True)

if __name__ == "__main__":
    main()
