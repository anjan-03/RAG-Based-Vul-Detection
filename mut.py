from transformers import pipeline

generator = pipeline("text-generation", model="t5-small")

prompt = "Modify this function while keeping the same vulnerability:\n"

vuln_function = "void unsafe(char *input) { char buffer[10]; strcpy(buffer, input); }"

output = generator(prompt + vuln_function, max_length=200)
print(output)