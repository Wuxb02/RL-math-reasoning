from .train import main as train_main
from .evaluate import main as evaluate_main
from .run_training import main as run_training_main
from .run_evaluation import main as run_evaluation_main

__all__ = ["train_main", "evaluate_main", "run_training_main", "run_evaluation_main"]
