import os
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = "/home/code871/Git/fsl-svm/server/FSL-dataset"

data = []

#loop through folders
for label in os.listdir(dataset_path):
  folder_path = os.path.join(dataset_path, label)

  if os.path.isdir(folder_path):
    for img_file in os.listdir(folder_path):
      img_path = os.path.join(folder_path, img_file)
      data.append([img_path, label])

#create dataframe
df = pd.DataFrame(data, columns=["filepath", "label"])

#split 80:20
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

#save CSV files
train_df.to_csv("/home/code871/Git/fsl-svm/server/csvs/train.csv", index=False)
test_df.to_csv("/home/code871/Git/fsl-svm/server/csvs/test.csv", index=False)

print("Done! CSV files created.")
