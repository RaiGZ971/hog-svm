import os
import pandas as pd

train_path = "./csvs/train.csv"
test_path = "./csvs/test.csv"

# load existing splits
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# normalize text (make category CAPS)
train_df["category"] = train_df["category"].astype(str).str.strip().str.upper()
test_df["category"] = test_df["category"].astype(str).str.strip().str.upper()

# filter the dataset to wanted category e
allowed = ["GREETING", "COLOR"]

train_df = train_df[train_df["category"].isin(allowed)].reset_index(drop=True)
test_df = test_df[test_df["category"].isin(allowed)].reset_index(drop=True)

# fix path format if needed
train_df["vid_path"] = train_df["vid_path"].str.replace("\\", os.sep)
test_df["vid_path"] = test_df["vid_path"].str.replace("\\", os.sep)

print("Filtered Train:", len(train_df))
print("Filtered Test", len(test_df))

print("\nTrain distribution:")
print(train_df["category"].value_counts())

print("\nTest distribution:")
print(test_df["category"].value_counts())

# save NEW filtered version
out_dir = "./csvs"

train_df.to_csv(os.path.join(out_dir, "train_filtered.csv"), index=False)
test_df.to_csv(os.path.join(out_dir, "test_filtered.csv"), index=False)

print("\nDone! Saved: ")
print("- train_filtered.csv")
print("- test_filtered.csv")
