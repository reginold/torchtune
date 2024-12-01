from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your model and config
model_path = "../output/hf_model_0003_0.pt"  # Adjust this to the correct checkpoint
config_path = "../output/config.json"
model_files_path="/import/mlcp-sc-nlp/llama-3_1/Meta-Llama-3.1-8B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=False)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_files_path,
                state_dict=torch.load(model_path, map_location="cpu"),
                    config=config_path,
                    )
model.eval()  # Set the model to evaluation mode

# Test input
test_input = "What is the capital of France?"

# Tokenize input
inputs = tokenizer(test_input, return_tensors="pt")

# Generate response
with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Response: {response}")

