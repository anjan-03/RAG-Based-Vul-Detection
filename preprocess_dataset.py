import numpy as np
import pandas as pd

df1 = pd.read_csv('data.csv')

X = df1.drop(columns=['target'])  # Replace 'target' with the actual target column name
y = df1['target']  # Replace 'target' with the actual target column name

# Separate the majority and minority classes
majority_class = y.value_counts().idxmax()
minority_class = y.value_counts().idxmin()

majority_class_data = df1[df1['target'] == majority_class]
minority_class_data = df1[df1['target'] == minority_class]

# Randomly downsample the majority class to match the minority class
majority_class_downsampled = majority_class_data.sample(n=minority_class_data.shape[0], random_state=42)

# Combine the downsampled majority class with the minority class
undersampled_df = pd.concat([majority_class_downsampled, minority_class_data])

# Shuffle the dataset to mix both classes properly
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

undersampled_df.to_csv('undersampled_data.csv', index=False)

df = pd.read_csv("undersampled_data.csv")
df = df[['func', 'target', 'cwe']]
df['func'] = df['func'].str.lower()
df['target'] = df['target'].astype(int)
df['cwe'] = df['cwe'].astype(object)
for idx in df[df['target'] == 0].index:
    df.at[idx, 'cwe'] = []
df.to_csv("preprocessed_data.csv", index=False)

print('Size of original dataset: ', df1.shape)
print('Size of undersampled dataset: ', df.shape)
