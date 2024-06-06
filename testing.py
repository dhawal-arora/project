import os
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load environment variables
dotenv.load_dotenv()

# Check if CUDA and MPS are available
print(torch.cuda.is_available())
print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")  
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Get Hugging Face access token from environment variables
access_token = os.getenv("HUGGING_FACE")

# Specify the model
model_name = "meta-llama/Llama-2-7b"  # Update to a known, valid model

try:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=access_token)

    # Initialize the text generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=device if device == "mps" else 0,
    )

    # Example usage
    text = "Once upon a time"
    generated_text = text_pipeline(text, max_length=50, num_return_sequences=1)
    print(generated_text)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the model name and your access permissions.")
