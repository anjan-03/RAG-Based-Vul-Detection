import pandas as pd
from datasets import Dataset
import torch

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df1 = pd.read_csv('data.csv')

# X = df1.drop(columns=['target'])  # Replace 'target' with the actual target column name
# y = df1['target']  # Replace 'target' with the actual target column name

# # Separate the majority and minority classes
# majority_class = y.value_counts().idxmax()
# minority_class = y.value_counts().idxmin()

# majority_class_data = df1[df1['target'] == majority_class]
# minority_class_data = df1[df1['target'] == minority_class]

# # Randomly downsample the majority class to match the minority class
# majority_class_downsampled = majority_class_data.sample(n=minority_class_data.shape[0], random_state=42)

# # Combine the downsampled majority class with the minority class
# undersampled_df = pd.concat([majority_class_downsampled, minority_class_data])

# # Shuffle the dataset to mix both classes properly
# undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# undersampled_df.to_csv('undersampled_data.csv', index=False)

df = pd.read_csv('undersampled_data.csv')
# print('Size of original dataset: ', df1.shape)
# print('Size of undersampled dataset: ', df.shape)

df['input_text'] = df['target'].fillna('').astype(str) + " " + df['cwe'].fillna('')  # Concatenate target and cwe as input
df['output_text'] = df['func']  # Output columns

# S1electing necessary columns for training
df = df[['input_text', 'output_text']]

# Convert DataFrame to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize_function(examples):
    # Ensure that 'input_text' is a list of strings, handling cases where it's not
    inputs = [str(text) if text is not None else "" for text in examples['input_text']]
    
    # Ensure that 'output_text' is a list of strings, handling cases where it's not
    labels = [str(text) if text is not None else "" for text in examples['output_text']]
    
    # Tokenize the input text
    input_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    
    # Tokenize the output text (labels)
    label_encodings = tokenizer(labels, truncation=True, padding="max_length", max_length=128)
    
    # Add the labels (from the tokenized output_text) to the inputs
    input_encodings['labels'] = label_encodings['input_ids']
    
    return input_encodings

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets (80% train, 20% validation)
train_dataset = tokenized_datasets.shuffle(seed=42).select([i for i in range(int(len(tokenized_datasets)*0.8))])
val_dataset = tokenized_datasets.shuffle(seed=42).select([i for i in range(int(len(tokenized_datasets)*0.8), len(tokenized_datasets))])

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save the model
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=8,   # Batch size per device during training
    per_device_eval_batch_size=16,   # Batch size per device during evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir="./logs",            # Directory to save logs
    logging_steps=10,
    eval_strategy="epoch",     # Evaluate every epoch
    save_strategy="epoch",           # Save model every epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # Pre-trained T5 model
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=val_dataset,            # Validation dataset
    tokenizer=tokenizer,                 # Tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_t5")
tokenizer.save_pretrained("./fine_tuned_t5")

# Evaluate the model
results = trainer.evaluate()
print(results)

# # Generate predictions using the fine-tuned model
# def generate_vulnerability(cwe_text):
#     # Tokenize the CWE input
#     inputs = tokenizer(cwe_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
#     # Generate the output (vulnerability description)
#     outputs = model.generate(inputs['input_ids'].to(device), max_length=128)
    
#     # Decode the output
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return prediction

# # Test the model with a CWE description
# sample_cwe = "CWE-79: Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')"
# print(generate_vulnerability(sample_cwe))  # Predict the vulnerability description based on the CWE