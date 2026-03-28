import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import visualize_results


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to results JSON file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for report"
    )

    args = parser.parse_args()

    visualize_results(args.input, args.output)


if __name__ == "__main__":
    main()
