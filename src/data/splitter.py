"""Dataset splitting utilities for train/validation/test splits."""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from src.utils.logger import get_logger

log = get_logger(__name__)


class DatasetSplitter:
    """Split datasets into train/validation/test sets."""

    @staticmethod
    def split(
        dataset: Dataset,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
        stratify_column: str | None = None,
    ) -> DatasetDict:
        """Split a dataset into train, validation, and test sets.

        Args:
            dataset: Dataset to split.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
            test_ratio: Fraction for test set.
            seed: Random seed for reproducibility.
            stratify_column: Optional column name for stratified splitting.

        Returns:
            DatasetDict with 'train', 'validation', 'test' keys.

        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid.
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
            raise ValueError("Split ratios must be non-negative")

        if train_ratio == 0:
            raise ValueError("Training ratio must be greater than 0")

        log.info(
            "splitting_dataset",
            num_samples=len(dataset),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        n = len(dataset)
        if n < 3:
            raise ValueError(f"Dataset too small to split: {n} samples (need at least 3)")

        # If stratified splitting is requested and column exists
        if stratify_column and stratify_column in dataset.column_names:
            return DatasetSplitter._stratified_split(
                dataset, train_ratio, val_ratio, test_ratio, seed, stratify_column
            )

        # Calculate split sizes
        val_test_ratio = val_ratio + test_ratio
        if val_test_ratio == 0:
            # All training
            return DatasetDict({"train": dataset, "validation": Dataset.from_list([]), "test": Dataset.from_list([])})

        # First split: train vs (val + test)
        split1 = dataset.train_test_split(test_size=val_test_ratio, seed=seed)
        train_ds = split1["train"]

        # Second split: val vs test from the remainder
        if test_ratio == 0:
            val_ds = split1["test"]
            test_ds = Dataset.from_list([])
        elif val_ratio == 0:
            val_ds = Dataset.from_list([])
            test_ds = split1["test"]
        else:
            relative_test = test_ratio / val_test_ratio
            split2 = split1["test"].train_test_split(test_size=relative_test, seed=seed)
            val_ds = split2["train"]
            test_ds = split2["test"]

        result = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        })

        log.info(
            "dataset_split_complete",
            train=len(result["train"]),
            validation=len(result["validation"]),
            test=len(result["test"]),
        )
        return result

    @staticmethod
    def _stratified_split(
        dataset: Dataset,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
        stratify_column: str,
    ) -> DatasetDict:
        """Perform stratified splitting based on a label column."""
        log.info("performing_stratified_split", column=stratify_column)

        # Group indices by label
        label_indices: dict[str, list[int]] = {}
        for i in range(len(dataset)):
            label = str(dataset[i][stratify_column])
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(i)

        import random

        rng = random.Random(seed)
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []

        for label, indices in label_indices.items():
            rng.shuffle(indices)
            n = len(indices)
            n_train = max(1, round(n * train_ratio))
            n_val = round(n * val_ratio)
            # Ensure at least 1 in train
            n_test = n - n_train - n_val

            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train : n_train + n_val])
            test_idx.extend(indices[n_train + n_val :])

        result = DatasetDict({
            "train": dataset.select(train_idx) if train_idx else Dataset.from_list([]),
            "validation": dataset.select(val_idx) if val_idx else Dataset.from_list([]),
            "test": dataset.select(test_idx) if test_idx else Dataset.from_list([]),
        })

        log.info(
            "stratified_split_complete",
            train=len(result["train"]),
            validation=len(result["validation"]),
            test=len(result["test"]),
            num_labels=len(label_indices),
        )
        return result
