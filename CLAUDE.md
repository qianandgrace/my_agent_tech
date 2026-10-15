# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This repository uses a conda environment `my_agent_tech` located at `C:\project\envs\my_agent_tech\python.exe`. Pip installation requires the Tsinghua mirror due to network issues in China: `pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple`. Required environment variables are defined in `.env`: `LAOZHANG_API_KEY`/`LAOZHANG_BASE_URL` for text generation, `QWEN_API_KEY`/`QWEN_BASE_URL` for image generation.

## Running Scripts

For test execution, use the environment's Python: `"C:\project\envs\my_agent_tech\python.exe" -m pytest <path_to_test>` or `python -m pytest <path_to_test>` if activated.

## Project Structure

This repository is an agent techniques practice collection organized into:

- `run_scripts/` - Self-contained agent examples demonstrating single concepts (e.g., LangGraph workflows, tool calling). Follow the pattern of `first_simple_agent.py` with `TypedDict` states and pure functions for nodes.
- `memories/` - Multi-turn conversation and history management examples.
- `movies_create_agent/` - Specialized video creation modules (currently contains a GIF generator). These deviate from the `run_scripts` pattern by using MaaS gateways and image processing.

## Key Architectural Patterns

### LangGraph StateGraph Workflows
When creating new agents:
- Use `TypedDict` for state definition
- Each node should be a pure function that accepts a state dict and returns partial updates
- Compile with `workflow.compile()` and invoke with state dict
- Include `if __name__ == "__main__":` entry points for standalone execution
- Follow the naming: `classification_node`, `entity_extraction`, etc.

### Model Integration
- Text generation: Always use `ChatOpenAI` with `gpt-4o-mini` via LAOZHANG endpoints
- Image generation: Use `requests` to call QWEN's MaaS gateway (`POST /chat/completions` with `wan2.7-image` model, content must be a list, response contains signed OSS URLs)

### Testing
- Write unit tests for pure utility functions (not API calls)
- Use pytest with TDD approach: tests first, then minimal implementation
- Test-only logic (like prompt parsing) should be in dedicated `test_<name>.py` files

## Special Considerations

For image generation tasks with QWEN endpoints:
- Do not use the standard OpenAI `images.generate` endpoint—it returns 400 errors
- The gateway requires list format content: `{"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]}`
- Response parsing extracts URLs from `output.choices[0].message.content[0]["image"]`
- Large images (2048x2048) require local resizing with PIL before GIF assembly

When adding new examples, follow the existing pattern: numbered README entries, snake_case filenames, and clear separation of concepts. Each script should demonstrate exactly one concept and include executable main blocks.