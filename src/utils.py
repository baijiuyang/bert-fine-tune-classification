import random
from collections import defaultdict

from datasets import load_dataset, Dataset


def load_local_dataset_with_hf(
    json_file_path: str,
    categories: list,
    total_sample_size: int,
    sentence_key: str = "headline",
    label_key: str = "category",
):
    """
    Loads a local JSON dataset with Hugging Face `datasets`, 
    filters by specified categories, and samples evenly among them.
    """

    if not categories:
        raise ValueError("The `categories` list cannot be empty.")

    # Load the dataset
    dataset = load_dataset("json", data_files=json_file_path, split="train")

    # Remove extra columns
    all_columns = dataset.column_names
    columns_to_remove = [
        col for col in all_columns if col not in [sentence_key, label_key]
    ]
    dataset = dataset.remove_columns(columns_to_remove)

    # Filter categories
    dataset = dataset.filter(lambda x: x[label_key] in categories)

    data = list(dataset)

    # Group rows by category
    data_by_cat = defaultdict(list)
    for row in data:
        data_by_cat[row[label_key]].append(row)

    # Calculate the sample size per category
    sample_per_category = max(1, total_sample_size // len(categories))

    sampled_data = []
    for cat in categories:
        cat_entries = data_by_cat[cat]
        if not cat_entries:
            continue
        chosen = random.sample(cat_entries, min(sample_per_category, len(cat_entries)))
        sampled_data.extend(chosen)

    # Convert back to a HF Dataset
    sampled_dataset = Dataset.from_list(sampled_data)

    return sampled_dataset
