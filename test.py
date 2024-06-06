import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import dotenv
from transformers import AutoTokenizer
import transformers
import torch
from transformers import pipeline
dotenv.load_dotenv()
from transformers import AutoTokenizer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print(torch.cuda.is_available())
print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")  
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", token= os.getenv("HUGGING_FACE"))
# print(pipe)

# langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# print(langchain_api_key)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

# model = ChatOpenAI(model="gpt-3.5-turbo")

# store = {}

# with_message_history = RunnableWithMessageHistory(model, get_session_history)
# config = {"configurable": {"session_id": "abc2"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Bob I'm 10")],
#     config=config,
# )

# print(response.content)
# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config,
# )
# print(response.content)
# response = with_message_history.invoke(
#     [HumanMessage(content="What's my age?")],
#     config=config,
# )
# print(response.content)


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