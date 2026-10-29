"""无 Key 可运行的测试。

运行方式（在项目根目录）：
  python tests/test_react.py
或
  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import MockLLM
from app.rag.retriever import BM25Index
from app.react_loop import run_agent
from app.tools import calculator


def test_calculator_basic():
    assert calculator("3*(4+5)") == {"result": 27}
    assert calculator("10/4") == {"result": 2.5}


def test_calculator_rejects_code():
    r = calculator("__import__('os').system('dir')")
    assert "error" in r


def test_bm25_search():
    chunks = [
        "MCP是模型上下文协议，标准化工具接入方式。",
        "ReAct是一种思考-行动-观察的循环。",
        "RAG是检索增强生成，用于注入知识。",
    ]
    idx = BM25Index(chunks)
    res = idx.search("MCP是什么")
    assert res and res[0]["index"] == 0, res


def test_react_mock_loop_terminates():
    """Mock 模式下：首轮工具调用 + 次轮最终答案，循环必然终止。"""
    llm = MockLLM()
    answer, trace = run_agent("1+1等于几", llm, max_steps=5)
    assert "2" in answer
    assert len(trace) == 2, trace


def test_react_max_steps_guard():
    """极端情况下循环必须被最大步数截断，不死循环。"""
    class DumbLLM:
        def chat(self, messages, tools=None, temperature=0.2, max_tokens=None):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "get_current_time", "arguments": "{}"},
                    }
                ],
            }

    answer, trace = run_agent("测试", DumbLLM(), max_steps=3)
    assert "最大步数" in answer
    assert len(trace) == 3


if __name__ == "__main__":
    tests = [
        test_calculator_basic,
        test_calculator_rejects_code,
        test_bm25_search,
        test_react_mock_loop_terminates,
        test_react_max_steps_guard,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
