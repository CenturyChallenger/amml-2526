import torch
from collections import Counter

test_path = "/src/data/test_dataset.pt"
holdout_path = "/src/data/holdout_dataset.pt"

# Load both datasets
train_data = torch.load(test_path, weights_only=False)
holdout_data = torch.load(holdout_path, weights_only=False)

def get_label_distribution(dataset, name="Dataset"):
    labels = [dataset[i][1] for i in range(len(dataset))]
    label_counts = Counter(labels)
    print(f"\n{name} Label Distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(dataset)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")

get_label_distribution(train_data, "Train Subset (6000)")
get_label_distribution(holdout_data, "Holdout Subset (3423)")
