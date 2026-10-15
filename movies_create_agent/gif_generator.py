"""GIF 动画生成 agent：一句话描述 → LangGraph 流水线 → 5 帧 GIF 文件。

流水线: 角色描述 → 5 步剧情 → 每帧图像 prompt → wan2.7 文生图 → PIL 拼装 GIF
文本节点用 gpt-4o-mini (LAOZHANG)，图像用 QWEN_API_KEY 对应的 MaaS 网关。
图像网关把文生图包装成 chat/completions 接口：content 传文本列表，响应里
output.choices[0].message.content 带图片 URL。

环境变量: LAOZHANG_API_KEY / LAOZHANG_BASE_URL（文本），QWEN_API_KEY / QWEN_BASE_URL（图像）
跑法: python movies_create_agent/gif_generator.py
"""
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from PIL import Image

load_dotenv()

IMAGE_MODEL = "wan2.7-image"
FRAME_COUNT = 5
FRAME_MS = 1000
FRAME_MAX_SIZE = 512
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

QWEN_BASE_URL = (os.getenv("QWEN_BASE_URL") or "").rstrip("/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY") or ""

llm = ChatOpenAI(
    api_key=os.getenv("LAOZHANG_API_KEY"),  # type: ignore
    base_url=os.getenv("LAOZHANG_BASE_URL"),
    model="gpt-4o-mini",
)


def parse_numbered_prompts(text: str, expected: int = FRAME_COUNT) -> List[str]:
    """从 LLM 输出中解析编号列表（支持 1. / 1) / **1.**），数量不符则抛 ValueError。"""
    prompts = []
    for line in text.split("\n"):
        m = re.match(r"^\*{0,2}(\d+)\s*[.)]\s*\*{0,2}\s*(.*)$", line.strip())
        if m:
            prompts.append(m.group(2).strip())
    if len(prompts) != expected:
        raise ValueError(f"Expected {expected} prompts, got {len(prompts)}. Please try again.")
    return prompts


def extract_image_url(data: dict) -> str | None:
    """从 MaaS 网关 chat/completions 响应里提取图片 URL。"""
    for choice in data.get("output", {}).get("choices", []):
        for part in choice.get("message", {}).get("content", []):
            if part.get("type") == "image" and part.get("image"):
                return part["image"]
    return None


def assemble_gif(frame_bytes: List[bytes], frame_ms: int = FRAME_MS, max_size: int = FRAME_MAX_SIZE) -> bytes | None:
    """把一列图片字节拼成循环播放的 GIF 字节；大图按比例缩到 max_size 以内。"""
    if not frame_bytes:
        return None
    images = []
    for b in frame_bytes:
        img = Image.open(io.BytesIO(b))
        img.thumbnail((max_size, max_size))
        images.append(img)
    buf = io.BytesIO()
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
    )
    return buf.getvalue()


class State(TypedDict):
    query: str
    character_description: str
    plot: str
    image_prompts: List[str]
    frame_bytes: List[bytes]
    gif_path: str


def generate_character_description(state: State):
    """生成主角/场景的详细描述，作为后续所有帧保持一致性的锚点。"""
    query = state["query"]
    response = llm.invoke([HumanMessage(content=(
        f"Based on the query '{query}', create a detailed description of the main "
        "character, object, or scene. Include specific details about appearance, "
        "characteristics, and any unique features. This description will be used "
        "to maintain consistency across multiple images."
    ))])
    return {"character_description": response.content}


def generate_plot(state: State):
    """生成 FRAME_COUNT 步的剧情，每步对应一帧。"""
    response = llm.invoke([HumanMessage(content=(
        f"Create a short, {FRAME_COUNT}-step plot for a GIF based on this query: "
        f"'{state['query']}' and featuring this description: "
        f"{state['character_description']}. Each step should be a brief description "
        "of a single frame, maintaining consistency throughout. Keep it "
        "family-friendly and avoid any sensitive themes."
    ))])
    return {"plot": response.content}


def generate_image_prompts(state: State):
    """为每一帧生成具体的图像 prompt，每条都带上角色描述以保持一致性。"""
    response = llm.invoke([HumanMessage(content=(
        f"Based on this plot: '{state['plot']}' and featuring this description: "
        f"{state['character_description']}, generate {FRAME_COUNT} specific, "
        "family-friendly image prompts, one for each step. Each prompt should be "
        "detailed enough for image generation, maintaining consistency, and "
        "suitable for a text-to-image model.\n\n"
        "Always include the following in EVERY prompt to maintain consistency:\n"
        "1. A brief reminder of the main character or object's key features\n"
        "2. The specific action or scene described in the plot step\n"
        "3. Any relevant background or environmental details\n\n"
        "Format each prompt as a numbered list item, like this:\n"
        "1. [Your prompt here]\n"
        "2. [Your prompt here]\n"
        "... and so on."
    ))])
    return {"image_prompts": parse_numbered_prompts(response.content)}


def generate_image(prompt: str) -> bytes | None:
    """调 MaaS 网关的 chat/completions 生成一张图，下载并返回图片字节。"""
    try:
        resp = requests.post(
            f"{QWEN_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
            json={
                "model": IMAGE_MODEL,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            },
            timeout=180,
        )
        resp.raise_for_status()
        url = extract_image_url(resp.json())
        if not url:
            print(f"No image in response for prompt: {prompt[:60]}...")
            return None
        img = requests.get(url, timeout=120)
        img.raise_for_status()
        return img.content
    except Exception as e:
        print(f"Frame generation failed: {e}")
        return None


def create_images(state: State):
    """并行为每条 prompt 生成一张图。"""
    with ThreadPoolExecutor(max_workers=FRAME_COUNT) as executor:
        frames = list(executor.map(generate_image, state["image_prompts"]))
    return {"frame_bytes": frames}


def create_gif(state: State):
    """把成功的帧拼成 GIF 并保存到 output/ 目录。"""
    frames = [b for b in state["frame_bytes"] if b]
    gif = assemble_gif(frames)
    if gif is None:
        return {"gif_path": ""}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S") + ".gif")
    with open(path, "wb") as f:
        f.write(gif)
    return {"gif_path": path}


workflow = StateGraph(State)
workflow.add_node("generate_character_description", generate_character_description)
workflow.add_node("generate_plot", generate_plot)
workflow.add_node("generate_image_prompts", generate_image_prompts)
workflow.add_node("create_images", create_images)
workflow.add_node("create_gif", create_gif)
workflow.set_entry_point("generate_character_description")
workflow.add_edge("generate_character_description", "generate_plot")
workflow.add_edge("generate_plot", "generate_image_prompts")
workflow.add_edge("generate_image_prompts", "create_images")
workflow.add_edge("create_images", "create_gif")
workflow.add_edge("create_gif", END)
app = workflow.compile()


if __name__ == "__main__":
    query = "A cat wearing a top hat and monocle, sitting at a desk and writing a letter with a quill pen."
    result = app.invoke({"query": query})

    print("Character/Scene Description:")
    print(result["character_description"])
    print("\nPlot:")
    print(result["plot"])
    print("\nImage Prompts:")
    for i, prompt in enumerate(result["image_prompts"], 1):
        print(f"{i}. {prompt}")
    ok = sum(1 for b in result["frame_bytes"] if b)
    print(f"\nFrames generated: {ok}/{len(result['frame_bytes'])}")
    print("\nGIF:", result["gif_path"] or "FAILED (no frames generated)")
