"""LangChain 重构版：用框架替代手写循环，同一组工具。

对比点（面试考点）：
- 框架替你做了：循环控制、消息拼接、工具结果解析、停止条件
- 框架隐藏的：异常兜底细节（handle_parsing_errors=True 才帮你兜）、上下文管理策略

安装：pip install -r requirements-framework.txt
运行（在项目根目录）：python langchain_version/agent_lc.py
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app import config
from app.rag.retriever import search_chunks
from app.tools import calculator as safe_calculator


@tool
def search_docs(query: str) -> str:
    """在本地知识库中检索与问题相关的文档片段（RAG）。"""
    return json.dumps(search_chunks(query), ensure_ascii=False)


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '3*(4+5)'。"""
    return json.dumps(safe_calculator(expression), ensure_ascii=False)


PROMPT = PromptTemplate.from_template(
    "你是运行在本地的最小 Agent，使用中文回答。\n"
    "可用工具：{tools}\n工具名称：{tool_names}\n{agent_scratchpad}"
)


def main():
    if not config.API_KEY:
        print("未配置 API_KEY，请先在 .env 中配置（参考 .env.example）。")
        return
    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0.2,
    )
    tools = [search_docs, get_current_time, calculator]
    agent = create_react_agent(llm, tools, PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=config.MAX_STEPS,
        handle_parsing_errors=True,
    )
    while True:
        q = input("提问（输入 q 退出）：").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        result = executor.invoke({"input": q})
        print("最终答案：", result["output"])


if __name__ == "__main__":
    main()
