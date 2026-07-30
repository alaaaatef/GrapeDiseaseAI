"""
Main Pipeline
Grape Disease Classification Project

This script runs the complete project pipeline.
"""

# ==================================================
# Imports
# ==================================================

from validation import validate_dataset
from dataloader import load_datasets
from utils import print_header
from preprocessing import preprocess_dataset
from visualization import visualize_dataset
# ==================================================
# Main Pipeline
# ==================================================

def main():

    print_header("GRAPE DISEASE CLASSIFICATION")

    # ---------------------------------------------
    # Step 1: Validate Dataset
    # ---------------------------------------------
    print("Running dataset validation...\n")
    validate_dataset()

    # ---------------------------------------------
    # Step 2: Load Dataset
    # ---------------------------------------------
    print("\nLoading datasets...\n")

    train_dataset, validation_dataset, test_dataset = load_datasets()

    # ---------------------------------------------
    # Step 3: Preprocess Dataset
    # ---------------------------------------------
    print("\nApplying preprocessing...\n")

    train_dataset = preprocess_dataset(
        train_dataset,
        training=True
    )

    validation_dataset = preprocess_dataset(
        validation_dataset,
        training=False
    )

    test_dataset = preprocess_dataset(
        test_dataset,
        training=False
    )

    print("Preprocessing completed successfully.\n")

    print("Visualizing dataset...\n")

    visualize_dataset(train_dataset)

    print("Visualization completed successfully.\n")

    # ---------------------------------------------
    # Step 4: Print Dataset Information
    # ---------------------------------------------
    print("Datasets loaded successfully.\n")

    print(f"Training batches   : {len(train_dataset)}")
    print(f"Validation batches : {len(validation_dataset)}")
    print(f"Testing batches    : {len(test_dataset)}")

    print("\nPipeline completed successfully.")

# ==================================================
# Run Pipeline
# ==================================================

if __name__ == "__main__":
    main()