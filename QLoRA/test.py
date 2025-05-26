from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_t5_with_qlora"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# If CUDA is available, use it for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate the model output based on open-ended question
def answer_question(query):
    input_text = f"Question: {query}"

    # Tokenize the input question
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move the tensors to the same device as the model (GPU if available, else CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the answer (the model will predict the response)
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'], 
            max_length=256,  # Increased max_length
            num_beams=3,     # Using fewer beams for better diversity
            temperature=1.2,  # Higher temperature for more variation
            top_k=40,        # Adjusting top_k
            top_p=0.9,       # Slightly reduced p-value
            do_sample=True,  # Enable sampling
            early_stopping=True
        )

    # Decode the output (convert token IDs back to text)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

# Example test cases (you can ask open-ended questions)
test_data = [
    "What is the function for CWE-703?"
]

# Test the model with some input queries
for query in test_data:
    print(f"Query: {query}")
    generated_output = answer_question(query)
    print("Answer:", generated_output)
    print("="*50)
