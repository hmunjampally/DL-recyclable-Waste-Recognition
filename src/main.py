from config import Config
from data_handlers.dataset import DatasetManager
from models.classifier import GarbageClassifier
from utils.environment import setup_environment
from utils.visualization import process_training_results, plot_training_results

def main():
    """Main execution function."""
    try:
        # Setup environment (this will clone YOLOv5 if needed)
        setup_environment()

        # Initialize configuration
        config = Config()

        # Create small dataset
        dataset_manager = DatasetManager(config)
        data_yaml = dataset_manager.create_small_dataset(
            config.ORIGINAL_DATASET,
            config.SMALL_DATASET,
            config.SAMPLES_PER_SPLIT
        )

        # Initialize and train classifier
        classifier = GarbageClassifier(config)
        results = process_training_results(
            classifier,
            str(data_yaml),
            epochs_list=[config.DEFAULT_EPOCHS]
        )

        # Plot results
        plot_training_results(results)

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
