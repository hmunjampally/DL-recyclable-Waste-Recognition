from config import Config
from data_handlers.dataset import DatasetManager
from models.classifier import GarbageClassifier
from utils.environment import setup_environment
from utils.visualization import process_training_results, plot_training_results
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """Main execution function."""
    try:
        logging.info("Setting up environment...")
        setup_environment()

        logging.info("Initializing configuration...")
        config = Config()

        '''# Create small dataset
        dataset_manager = DatasetManager(config)
        data_yaml = dataset_manager.create_small_dataset(
            config.ORIGINAL_DATASET,
            config.SMALL_DATASET,
            config.SAMPLES_PER_SPLIT
        )'''
        
        
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
        plot_training_results(results)
        
        logging.info("Training completed successfully!")


    except Exception as e:
        logging.error("Error in main execution", exc_info=True)

if __name__ == "__main__":
    main()
