"""工具定义：Function Calling 的 schema + 执行函数。

安全说明：calculator 用 AST 白名单求值，禁止任意代码执行。
"""
import ast
import datetime
import operator

from .rag.retriever import search_chunks

# ---------------------------------------------------------------- 执行函数

def get_current_time():
    """返回当前时间字符串。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
        v = _safe_eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_OPS:
            raise ValueError("不支持的运算符")
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("表达式包含不允许的语法")


def calculator(expression: str):
    """安全计算算术表达式，例如 '3*(4+5)'。"""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree)}
    except Exception as e:  # 任何异常都转成工具返回，不中断 Agent
        return {"error": f"无法计算: {e}"}


# ---------------------------------------------------------------- 工具 schema 与注册表

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在本地知识库中检索与问题相关的文档片段（RAG）。当问题涉及知识库内容时使用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索关键词/问题"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如 '3*(4+5)'。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "算术表达式"}},
                "required": ["expression"],
            },
        },
    },
]

# 工具名白名单：名字与执行函数一一对应，查不到即拒绝（生产兜底之一）
EXECUTORS = {
    "search_docs": lambda query: search_chunks(query),
    "get_current_time": get_current_time,
    "calculator": calculator,
}
