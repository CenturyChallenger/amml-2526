import torch

def inspect_pt_file(file_path):
    data = torch.load(file_path, weights_only=False)
    print(f"Type of loaded object: {type(data)}")
    print(f"Object keys (if dict): {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    
    # If it's a Dataset or Subset
    if hasattr(data, '__len__'):
        print(f"Length (number of samples): {len(data)}")
    
    # If it's a DataLoader
    if hasattr(data, 'dataset'):
        print(f"DataLoader dataset length: {len(data.dataset)}")
    
    # Try to see the first sample
    try:
        if hasattr(data, '__getitem__'):
            sample = data[0]
            print(f"First sample type: {type(sample)}")
            if isinstance(sample, (tuple, list)):
                print(f"First sample structure: {[type(x).__name__ for x in sample]}")
                if hasattr(sample[0], 'shape'):
                    print(f"Input shape: {sample[0].shape}")
                if hasattr(sample[1], 'shape'):
                    print(f"Label shape: {sample[1].shape}")
    except Exception as e:
        print(f"Could not get first sample: {e}")
    
    return data

# Usage
#model_path = "/media/philip/4eb935af-a979-4ebf-9f37-60ba267292b8/PN Dropbox/Philip Nyamwaya/Family/Shai/UK Jobs/GTV/UoS/Advanced Methods in ML/Assignment/query_stuff/amml-2526-main/virtual_env/src/data/test_dataset.pt"
model_path = "/media/philip/4eb935af-a979-4ebf-9f37-60ba267292b8/PN Dropbox/Philip Nyamwaya/Family/Shai/UK Jobs/GTV/UoS/Advanced Methods in ML/Assignment/query_stuff/amml-2526-main/virtual_env/src/data/holdout_dataset.pt"
data = inspect_pt_file(model_path)

