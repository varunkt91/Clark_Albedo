

import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(input_path, output_dir, n_samples=None, 
                     train_ratio=0.4, val_ratio=0.3, test_ratio=0.3,
                     random_state=42, return_data=False):
    """
    Randomly split a CSV into train, validation, and test sets.
    
    Parameters:
        input_path (str): Path to the CSV file.
        output_dir (str): Directory to save train/val/test CSVs.
        n_samples (int or None): Number of rows to sample. If None, use all rows.
        train_ratio (float): Fraction for training set.
        val_ratio (float): Fraction for validation set.
        test_ratio (float): Fraction for test set.
        random_state (int): Random seed for reproducibility.
        return_data (bool): If True, also return the DataFrames.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train/Val/Test ratios must sum to 1.0")

    # Load data
    data = pd.read_csv(input_path)

    # Sample rows if n_samples is specified
    if n_samples is not None:
        n_samples = min(n_samples, len(data))
        data = data.sample(n=n_samples, random_state=random_state)

    # Split into train and temp
    train, temp = train_test_split(data, test_size=(1-train_ratio), random_state=random_state)

    # Split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(temp, test_size=(1 - val_size), random_state=random_state)

    # Save CSVs
    train.to_csv(f"{output_dir}/train_200k.csv", index=False)
    val.to_csv(f"{output_dir}/validation_200k.csv", index=False)
    test.to_csv(f"{output_dir}/test_200k.csv", index=False)

    print(f"✅ Files saved in {output_dir}")
    print(f"   Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

    if return_data:
        return train, val, test
