from pathlib import Path

class Config:
    """Configuration settings for the garbage classifier."""

    def __init__(self):
        # Get project root directory
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()

        # Paths
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.ORIGINAL_DATASET = self.DATA_DIR / "TrashNet"
        #self.SMALL_DATASET = self.DATA_DIR / "small_trashnet"
        self.RESULTS_DIR = self.PROJECT_ROOT / "training_results"

        # Ensure paths use correct separators for the current OS
        self.ORIGINAL_DATASET = Path(str(self.ORIGINAL_DATASET))
        #self.SMALL_DATASET = Path(str(self.SMALL_DATASET))
        self.RESULTS_DIR = Path(str(self.RESULTS_DIR))

        # Model settings
        self.BATCH_SIZE = 16
        self.IMAGE_SIZE = 640
        self.NUM_CLASSES = 6
        self.CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

        # Training settings
        self.DEFAULT_EPOCHS = 30
        self.PATIENCE = 10

        '''# Dataset settings
        self.SAMPLES_PER_SPLIT = {
            'train': 100,
            'val': 50,
            'test': 20
        }'''

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.DATA_DIR, self.RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
