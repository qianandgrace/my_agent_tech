---
name: adding-agent-examples
description: 在本仓库新增、整理或修改 agent 示例脚本时使用，覆盖 run_scripts/ 与 memories/ 的目录归属、文件命名、dotenv 模型接入、LangGraph 与 LangChain 的既定写法、运行方式和 README 编号更新约定。
whenToUse: 用户要求新增一个 agent 示例、练习脚本或 demo，把某段 agent 代码整理进本仓库，或重构现有的 run_scripts/、memories/ 脚本使其符合仓库约定时。
---

# 新增 agent 示例脚本

本仓库是 agent 技术练习集：一个主题一个可直接运行的 Python 脚本，README 用编号清单登记它们。新增内容时保持"能跑、够短、一眼看懂一个概念"。

## 目录归属

| 目录 | 放什么 |
|---|---|
| `run_scripts/` | 可独立运行的完整示例：单次调用、流程编排、工具调用、检索等 |
| `memories/` | 与上下文/记忆相关的示例：多轮会话、历史管理、状态持久化 |

不确定时放进 `run_scripts/`。不要新建顶层目录来放单个脚本。

## 命名

- 文件名用 snake_case，英文，描述主题而非序号：`react_agent_tools.py`、`rag_with_pgvector.py`。
- 序号只体现在 README 清单里，不进文件名。
- 一个文件只演示一个概念；需要两个概念就拆成两个文件，并在 README 相邻登记。

## 模型接入

所有示例统一用 dotenv + OpenAI 兼容端点，不要硬编码 key：

```python
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("LAOZHANG_API_KEY"),  # type: ignore
    base_url=os.getenv("LAOZHANG_BASE_URL"),
    model="gpt-4o-mini",
)
```

- `.env` 已在 `.gitignore` 中，**不要**把 key、token 或 base_url 的真实值写进代码、README、注释或提交信息。
- 需要新的环境变量时，代码里用 `os.getenv(...)` 读取，并把变量名（不含值）写进 README 的说明。
- 每个 import 只出现一次；现有 `run_scripts/first_simple_agent.py` 里重复 import 了 `ChatOpenAI`，新脚本不要照抄这个瑕疵。

## 两种骨架

### LangGraph 流程编排（`run_scripts/`）

用 `TypedDict` 定义 State，每个节点是 `state -> 部分更新 dict` 的纯函数，最后编译并调用：

```python
from typing import List, TypedDict

from langgraph.graph import END, StateGraph


class State(TypedDict):
    text: str
    result: str


def my_node(state: State):
    """一句话说明这个节点做什么。"""
    return {"result": state["text"].upper()}


workflow = StateGraph(State)
workflow.add_node("my_node", my_node)
workflow.set_entry_point("my_node")
workflow.add_edge("my_node", END)
app = workflow.compile()

if __name__ == "__main__":
    print(app.invoke({"text": "hello"}))
```

### 带历史的多轮会话（`memories/`）

用 `RunnableWithMessageHistory` + 显式 `session_id`，把历史存储抽成函数以便替换：

```python
chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

response = chain_with_history.invoke(
    {"input": "你好"},
    config={"configurable": {"session_id": "user_123"}},
)
print("AI:", response.content)
```

## 必须有的收尾

- 每个脚本都要有 `if __name__ == "__main__":` 入口，把示例输入和 `print` 结果放在里面，import 时不执行网络调用。
- 打印结果带前缀标签（如 `Classification:`、`AI:`），便于对照输出理解节点。
- 用 `templates/example_graph.py` 作为起点，删掉不需要的节点即可。

## 运行方式

仓库没有依赖清单文件，按脚本实际 import 安装：

```
pip install python-dotenv langchain-openai langgraph langchain-community
python run_scripts/your_script.py
```

确认 `.env` 存在且含 `LAOZHANG_API_KEY`、`LAOZHANG_BASE_URL`，否则脚本会在创建模型或首次调用时报认证错误——先排查这个，再怀疑代码逻辑。

## 更新 README

在 `README.md` 现有编号后追加一行，格式沿用现有条目的 `序号.描述, 相对路径`：

```
3.tool-calling agent, run_scripts/tool_calling_agent.py
```

不要重排已有编号，也不要改写已有条目。

## 交付前自检

- [ ] 文件在 `run_scripts/` 或 `memories/`，snake_case 命名
- [ ] `load_dotenv()` 在创建模型之前调用
- [ ] 无硬编码 key/token，`.env` 未进入暂存区
- [ ] 有 `if __name__ == "__main__":`，且 import 无副作用
- [ ] README 已追加编号条目
- [ ] 实际运行过一次并看到预期输出
