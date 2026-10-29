"""裸 Python 手写 ReAct 循环：思考 → 工具调用 → 观察 → 再思考 → 最终答案。

不依赖任何 Agent 框架。面试讲清"Agent 到底是什么"的最小实现：
- 状态：messages 列表（对话上下文）
- 工具：TOOLS schema + EXECUTORS 白名单
- 循环：LLM 输出工具调用 → 执行 → 结果回填 → 继续，直到无工具调用
- 停止条件：MAX_STEPS 上限 + 无工具调用即出最终答案
"""
import json

from . import config
from .tools import TOOLS, EXECUTORS


def _tool_descriptions():
    lines = []
    for t in TOOLS:
        f = t["function"]
        params = list(f.get("parameters", {}).get("properties", {}).keys())
        lines.append(f"- {f['name']}({', '.join(params)})：{f['description']}")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """你是运行在本地的 ReAct Agent，通过"思考→行动→观察"循环完成任务。

可用工具：
{tool_descriptions}

规则：
1. 需要外部信息或计算时，先调用工具；得到 Observation 后再继续思考。
2. 每次工具调用必须使用标准 Function Calling 格式（不要自己编格式）。
3. 调用工具最多 {max_steps} 次。得到足够信息后，直接输出最终答案（用"最终答案："开头），不要再调用工具。
4. 工具参数必须符合 schema。若工具返回 error，换一种方式重试。
5. 回答用中文，简洁；引用知识库内容时说明来源。"""


def build_system_prompt():
    return SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=_tool_descriptions(),
        max_steps=config.MAX_STEPS,
    )


def _truncate(text, limit=2000):
    """上下文截断：防止工具返回结果把上下文撑爆（Token 爆炸兜底之一）。"""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"...[已截断，共{len(text)}字符]"


def run_agent(question, llm, max_steps=None, verbose=False):
    """执行 ReAct 循环，返回 (final_answer, trace)。"""
    max_steps = max_steps or config.MAX_STEPS
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": question},
    ]
    trace = []
    for step in range(1, max_steps + 1):
        msg = llm.chat(messages, tools=TOOLS)
        messages.append(msg)
        trace.append({"step": step, "message": msg})

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            if answer:
                return answer, trace
            return "（模型未返回内容，且无工具调用，无法继续）", trace

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            # 工具名白名单校验
            if name not in EXECUTORS:
                observation = {"error": f"未知工具: {name}，可用工具: {list(EXECUTORS)}"}
            else:
                try:
                    observation = EXECUTORS[name](**args)
                except TypeError as e:
                    observation = {"error": f"工具参数错误: {e}"}
                except Exception as e:
                    observation = {"error": f"工具执行异常: {e}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"step{step}"),
                "content": _truncate(json.dumps(observation, ensure_ascii=False)),
            })
            if verbose:
                print(
                    f"[Step {step}] 工具 {name}({json.dumps(args, ensure_ascii=False)})"
                    f" -> {_truncate(json.dumps(observation, ensure_ascii=False), 300)}"
                )
    return "已达到最大步数，仍未得到最终答案（可尝试把问题拆小，或调整 MAX_STEPS）", trace
