import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset

# Step 1: Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_t5"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load validation data - Sample code to load validation dataset if it's available
df = pd.read_csv('preprocessed_data.csv') 
df['input_text'] = df['target'].fillna('').astype(str) + " " + df['cwe'].fillna('')
df['output_text'] = df['func']
df = df[['input_text', 'output_text']]
dataset = Dataset.from_pandas(df)
val_dataset = dataset.shuffle(seed=42).select([i for i in range(int(len(dataset)*0.9), len(dataset))])

# Step 3: Tokenize the validation data (same tokenization function used earlier)
def tokenize_function(examples):
    inputs = [str(text) if text is not None else "" for text in examples['input_text']]
    labels = [str(text) if text is not None else "" for text in examples['output_text']]
    
    input_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    label_encodings = tokenizer(labels, truncation=True, padding="max_length", max_length=128)
    input_encodings['labels'] = label_encodings['input_ids']
    
    return input_encodings

tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Step 4: Generate Predictions
def generate_predictions(dataset):
    predictions = []
    for batch in dataset:
        inputs = torch.tensor(batch['input_ids']).unsqueeze(0).to(device)  # Add batch dimension
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0).to(device)
        
        # Generate output tokens
        outputs = model.generate(inputs, attention_mask=attention_mask, max_length=128, num_beams=5, early_stopping=True)
        
        # Decode generated tokens
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded_output)
    
    return predictions

# Get predictions for the validation dataset
predictions = generate_predictions(tokenized_val_dataset)

# Step 5: Print out some of the predictions
for i in range(5):  # Print the first 5 predictions
    print(f"Input: {tokenized_val_dataset[i]['input_text']}")
    print(f"Predicted Output: {predictions[i]}")
    print("="*50)