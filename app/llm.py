"""LLM 客户端：OpenAI 兼容 chat/completions，支持 tools（Function Calling）。

- LLMClient：真实请求（DeepSeek / 通义百炼均兼容）
- MockLLM：MOCK_MODE=1 时使用，不发真实请求，用于无 Key 调试循环逻辑
"""
import json

import requests

from . import config


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or config.API_KEY
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.model = model or config.CHAT_MODEL
        if not self.api_key:
            raise ValueError(
                "缺少 API_KEY：请在项目根目录复制 .env.example 为 .env 并填入 Key。"
                "无 Key 调试请用 python -m app.agent \"问题\" --mock"
            )

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=None):
        """一次 chat 调用。返回 OpenAI 风格 message 字典（可能含 tool_calls）。"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM 请求失败 [{resp.status_code}]: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"]


class MockLLM:
    """无 Key 演示：按关键词路由到工具，验证循环与工具链路（不调真实模型）。"""

    def __init__(self):
        pass

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=None):
        last = messages[-1]
        # 上一步是工具结果 → 给出最终答案
        if last["role"] == "tool":
            obs = last["content"]
            return {
                "role": "assistant",
                "content": (
                    f"最终答案：根据工具返回结果：{obs[:120]}……\n"
                    "（当前为 Mock 演示模式，未调用真实模型。配置 API_KEY 后即为真实回答。）"
                ),
            }
        # 首次提问 → 按关键词路由到工具
        q = last.get("content", "")
        if "=" in q or any(c.isdigit() for c in q):
            name, args = "calculator", {"expression": "1+1"}
        elif any(k in q for k in ["MCP", "RAG", "Agent", "检索", "chunk", "幻觉", "工具"]):
            name, args = "search_docs", {"query": q}
        else:
            name, args = "get_current_time", {}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "mock-1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ],
        }
