from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the code documentation model
model_name = "SEBIS/code_trans_t5_base_code_documentation_generation_python"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Input code for summarization
input_code = """
def prime(a):
    s=a*2
    return s
"""

# Prompt for code documentation (simplified for T5)
input_text = f"generate documentation: {input_code}"

# Tokenize the input text
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding=True
)

# Generate the summary/documentation
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,          # Limit to a concise sentence
    min_length=10,          # Ensure a meaningful length
    num_beams=4,            # Beam search for quality
    length_penalty=1.0,     # Neutral length preference
    early_stopping=True
)

# Decode the output
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the results
print("\nGenerated Summary:")
print(summary)