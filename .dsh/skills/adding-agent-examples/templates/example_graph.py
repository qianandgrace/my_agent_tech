"""<一句话说明这个示例演示什么>。

模板骨架：复制到 run_scripts/ 后改名，删掉不需要的节点，替换 prompt 与示例输入。
"""

import os
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("LAOZHANG_API_KEY"),  # type: ignore
    base_url=os.getenv("LAOZHANG_BASE_URL"),
    model="gpt-4o-mini",
)


class State(TypedDict):
    text: str
    entities: List[str]
    summary: str


def extract_node(state: State):
    """从文本中抽取实体，返回逗号分隔列表。"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract all the entities (Person, Organization, Location) from the "
            "following text. Provide the result as a comma-separated list.\n\n"
            "Text:{text}\n\nEntities:"
        ),
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}


def summarize_node(state: State):
    """用一句话概括文本。"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    return {"summary": llm.invoke([message]).content.strip()}


workflow = StateGraph(State)
workflow.add_node("extract_node", extract_node)
workflow.add_node("summarize_node", summarize_node)
workflow.set_entry_point("extract_node")
workflow.add_edge("extract_node", "summarize_node")
workflow.add_edge("summarize_node", END)
app = workflow.compile()


if __name__ == "__main__":
    sample_text = "替换成这个示例要处理的文本。"
    result = app.invoke({"text": sample_text})
    print("Entities:", result["entities"])
    print("Summary:", result["summary"])
