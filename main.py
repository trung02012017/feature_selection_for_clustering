import argparse

from kmeans import run_kmeans
from ahc import run_ahc
from visualization import run_visualize_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        help="Select the mode 'visualize' or 'train' data to run",
                        choices=["visualize", "train"])
    parser.add_argument("--model",
                        help="Select model 'kmeans' or 'ahc' to train",
                        choices=["kmeans", "ahc"])
    args = parser.parse_args()

    if args.mode == "visualize":
        run_visualize_data()
    if args.mode == "train":
        if args.model == "kmeans":
            run_kmeans()
        if args.model == "ahc":
            run_ahc()