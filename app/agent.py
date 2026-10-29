"""命令行入口。

用法（在项目根目录）：
  python -m app.agent "什么是MCP？"          # 真实调用 LLM（需 .env 配置 API_KEY）
  python -m app.agent "1+1等于几" --mock     # 无 Key 演示
  python -m app.agent "问题" --max-steps 8   # 自定义最大步数
"""
import argparse

from . import config
from .llm import LLMClient, MockLLM
from .react_loop import run_agent


def main():
    parser = argparse.ArgumentParser(description="最小 RAG Agent（ReAct + Function Calling）")
    parser.add_argument("question", nargs="?", help="提问，例如：什么是MCP？")
    parser.add_argument("--mock", action="store_true", help="使用 MockLLM 无 Key 演示")
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        return

    use_mock = args.mock or config.MOCK_MODE
    llm = MockLLM() if use_mock else LLMClient()

    print(f"提问：{args.question}\n")
    answer, trace = run_agent(args.question, llm, max_steps=args.max_steps, verbose=True)
    print(f"\n最终答案：{answer}")


if __name__ == "__main__":
    main()
