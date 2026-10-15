import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatOpenAI(
        api_key=os.getenv("LAOZHANG_API_KEY"), # type: ignore
        base_url=os.getenv("LAOZHANG_BASE_URL"),
        model="gpt-4o-mini")

# create simple in-memory store
store = {}

def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

if __name__ == "__main__":
    session_id = "user_123"
    response1 = chain_with_history.invoke(
        {"input": "Hello! How are you?"},
        config={"configurable": {"session_id": session_id}}
    )
    print("AI:", response1.content)

    response2 = chain_with_history.invoke(
        {"input": "What was my previous message?"},
        config={"configurable": {"session_id": session_id}}
    )
    print("AI:", response2.content)