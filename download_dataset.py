import os
try:
    import kagglehub
    print("Kagglehub is available. Downloading datasets...")
    path1 = kagglehub.dataset_download("ryandpark/fruit-quality-classification")
    print("First dataset downloaded to:", path1)
    
    path2 = kagglehub.dataset_download("sebastiancos21/dataset-frutas")
    print("Second dataset downloaded to:", path2)
    
    # List contents
    print("\nRoot contents of first dataset:")
    for root, dirs, files in os.walk(path1):
        # Limit depth to 2
        level = root.replace(path1, '').count(os.sep)
        if level < 2:
            print(f"{'  ' * level}- {os.path.basename(root)}/ ({len(dirs)} dirs, {len(files)} files)")

    print("\nRoot contents of second dataset:")
    for root, dirs, files in os.walk(path2):
        # Limit depth to 2
        level = root.replace(path2, '').count(os.sep)
        if level < 2:
            print(f"{'  ' * level}- {os.path.basename(root)}/ ({len(dirs)} dirs, {len(files)} files)")
except Exception as e:
    print("Error:", e)

