import pandas as pd
from datasets import Dataset
import torch

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('preprocessed_data.csv')
df['input_text'] = df['target'].fillna('').astype(str) + " " + df['cwe'].fillna('')  # Concatenate target and cwe as input
df['output_text'] = df['func']  # Output columns

# Selecting necessary columns for training
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

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)

# QLoRA: Define the LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank for LoRA. You can experiment with this value
    lora_alpha=32,  # Alpha parameter to scale the LoRA weights
    lora_dropout=0.1,  # Dropout for LoRA
    task_type=TaskType.SEQ_2_SEQ_LM,  # This is for sequence-to-sequence tasks (like T5)
)

# Apply QLoRA to the model
model = get_peft_model(model, lora_config)

# Define training arguments (as before)
training_args = TrainingArguments(
    output_dir="./results",          
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir="./logs",            
    logging_steps=10,
    eval_strategy="epoch",     
    save_strategy="epoch",           
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    tokenizer=tokenizer,                 
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_t5_with_qlora")
tokenizer.save_pretrained("./fine_tuned_t5_with_qlora")

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
