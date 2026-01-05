import torch
from collections import Counter

test_path = "/media/philip/4eb935af-a979-4ebf-9f37-60ba267292b8/PN Dropbox/Philip Nyamwaya/Family/Shai/UK Jobs/GTV/UoS/Advanced Methods in ML/Assignment/query_stuff/amml-2526-main/virtual_env/src/data/test_dataset.pt"
holdout_path = "/media/philip/4eb935af-a979-4ebf-9f37-60ba267292b8/PN Dropbox/Philip Nyamwaya/Family/Shai/UK Jobs/GTV/UoS/Advanced Methods in ML/Assignment/query_stuff/amml-2526-main/virtual_env/src/data/holdout_dataset.pt"

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
