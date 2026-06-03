import os
try:
    import kagglehub
    print("Kagglehub is available. Downloading dataset...")
    path = kagglehub.dataset_download("ryandpark/fruit-quality-classification")
    print("Dataset downloaded to:", path)
    
    # List contents
    print("\nRoot contents of dataset:")
    for root, dirs, files in os.walk(path):
        # Limit depth to 2
        level = root.replace(path, '').count(os.sep)
        if level < 2:
            print(f"{'  ' * level}- {os.path.basename(root)}/ ({len(dirs)} dirs, {len(files)} files)")
except Exception as e:
    print("Error:", e)
