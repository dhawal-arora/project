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
from transformers import AutoTokenizer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

dotenv.load_dotenv()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
print(langchain_api_key)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}

with_message_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="hi")],
    config=config,
)

print(response.content)
response = with_message_history.invoke(
    [HumanMessage(content="im 20")],
    config=config,
)
print(response.content)
response = with_message_history.invoke(
    [HumanMessage(content="give me a fact about messi")],
    config=config,
)
print(response.content)

