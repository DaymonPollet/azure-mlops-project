import os
import argparse
import logging
from glob import glob
import math
import random

def main():
    """Main function of the script."""

    SEED = 42

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", help="All the datasets to combine")
    parser.add_argument("--training_data_output", type=str, help="path to training output data")
    parser.add_argument("--testing_data_output", type=str, help="path to testing output data")
    parser.add_argument("--split_size", type=int, help="Percentage to use as Testing data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.datasets)
    print("Training folder:", args.training_data_output)
    print("Testing folder:", args.testing_data_output)
    print("Split size:", args.split_size)

    train_test_split_factor = args.split_size / 100 # Alias
    datasets = args.datasets

    training_datapaths = []
    testing_datapaths = []

    for dataset in datasets:
        # Construct path robustly, assuming dataset is a URI/folder path that glob can process
        # The lab context implies datasets are mounted folders.
        animal_images = glob(os.path.join(dataset, "*.jpg"))
        print(f"Found {len(animal_images)} images for {dataset}")

        random.seed(SEED) # Use the same random seed as I use and defined in the earlier cells
        random.shuffle(animal_images) # Shuffle the data so it's randomized

        ## Testing images
        amount_of_test_images = math.ceil(len(animal_images) * train_test_split_factor) # Get a small percentage of testing images

        animal_test_images = animal_images[:amount_of_test_images]
        animal_training_images = animal_images[amount_of_test_images:]

        # Add them all to the other ones
        testing_datapaths.extend(animal_test_images)
        training_datapaths.extend(animal_training_images)

        # print(testing_datapaths[:5]) # Uncomment for debugging

        # Ensure output directories exist
        os.makedirs(args.testing_data_output, exist_ok=True)
        os.makedirs(args.training_data_output, exist_ok=True)

        # Write the data to the output folders
        for img_src_path in animal_test_images:
            # Open the img, which is a string filepath, then save it to the args.testing_data_output directory
            img_dest_path = os.path.join(args.testing_data_output, os.path.basename(img_src_path))
            with open(img_src_path, "rb") as f_src:
                with open(img_dest_path, "wb") as f_dest:
                    f_dest.write(f_src.read())
            # print(f"Copied {os.path.basename(img_src_path)} to test folder.") # Uncomment for debugging

        for img_src_path in animal_training_images:
            # Open the img, which is a string filepath, then save it to the args.training_data_output directory
            img_dest_path = os.path.join(args.training_data_output, os.path.basename(img_src_path))
            with open(img_src_path, "rb") as f_src:
                with open(img_dest_path, "wb") as f_dest:
                    f_dest.write(f_src.read())
            # print(f"Copied {os.path.basename(img_src_path)} to train folder.") # Uncomment for debugging

if __name__ == "__main__":
    main()