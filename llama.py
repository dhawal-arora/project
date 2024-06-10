#------------------------------IMPORTS-----------------------------------
import os
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import dotenv
from transformers import AutoTokenizer
import transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

dotenv.load_dotenv()

print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")  
device = torch.device('cpu')
print(f"Using device: {device}")

# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", token= os.getenv("HUGGING_FACE"))
# prompt = "Once upon a time"
# generated_text = pipe(prompt, max_length=1000,truncation=True ,num_return_sequences=1)
# print(generated_text[0]['generated_text'])

access_token = os.getenv("HUGGING_FACE")
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    model, 
    token=access_token
)

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("ready")
text = "Please tell me a fact about messi."
generated_text = pipeline(text, max_length=1000, truncation=True, num_return_sequences=1)
print(generated_text)
