# RAG + ReAct 最小 Agent

从零手写的 RAG 问答 Agent：**裸 Python 实现 ReAct 循环 + Function Calling + BM25 中文检索**，附 LangChain 重构版与 Streamlit 演示界面。用于 AI Agent 实习求职的个人项目。

## 为什么做这个项目

- 企业招 Agent 实习生看重"能跑通、能讲清"的端到端项目，而不是课程列表
- 项目**刻意用裸 Python 手写 ReAct 循环，不依赖框架**，把 Agent 机制（状态、工具、停止条件）讲透
- 中文语料用 **BM25 + jieba**：零成本、零向量库、可解释，为升级向量检索留了接口

## 技术栈

Python 3.13 · DeepSeek（OpenAI 兼容）/ 通义百炼 · requests · jieba BM25 · LangChain（可选）· Streamlit（可选）

## 架构

```
用户提问
   │
   ▼
┌──────────────────────────────┐
│  ReAct 循环（裸 Python）       │
│  思考 → 工具调用 → 观察 → …    │
│  停止条件：MAX_STEPS / 无工具调用 │
└─────────────┬────────────────┘
              │ Function Calling（LLM 输出结构化工具调用）
      ┌───────┴────────┐
      ▼                ▼
 search_docs     calculator / get_current_time
      │
      ▼
BM25 检索（jieba 分词，语料切片）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
copy .env.example .env   # Windows
# 编辑 .env：填入 API_KEY；LLM_PROVIDER=deepseek 或 qwen

# 3. 无 Key 演示（Mock 模式，验证循环逻辑）
python -m app.agent "什么是MCP？" --mock

# 4. 真实运行（需 API Key）
python -m app.agent "什么是MCP？"

# 5. 图形界面
pip install -r requirements-ui.txt
streamlit run ui/app.py
```

## 效果示例（Mock 模式实测输出）

```
> python -m app.agent "什么是MCP？" --mock

[Step 1] 工具 search_docs({"query": "什么是MCP？"})
  -> {"hits": 3, "results": [{"index": 8, "score": 3.1522, "text": "面试官常问...MCP 解决什么问题..."}]}

最终答案：根据工具返回结果：{"hits": 3, "results": [...]}
（当前为 Mock 演示模式，未调用真实模型。配置 API_KEY 后即为真实回答。）
```

> 单测：`python tests/test_react.py` → 5/5 passed（含安全计算、BM25 命中、循环终止、最大步数截断）。
> 接上真实 Key 后，把一次真实问答的完整输出贴在这里，作为演示截图素材。

## 目录结构

```
ai-agent-project/
├── app/
│   ├── config.py        # .env 配置（供应商/模型/RAG 参数/步数）
│   ├── llm.py           # LLM 客户端（DeepSeek/通义兼容）+ MockLLM
│   ├── tools.py         # 工具 schema + 执行函数（AST 白名单安全计算）
│   ├── react_loop.py    # ★ 裸 Python ReAct 循环（本项目核心）
│   ├── agent.py         # 命令行入口
│   └── rag/
│       ├── loader.py    # 文档加载与切片（chunk size/overlap 可调）
│       └── retriever.py # BM25 检索（jieba 分词）
├── langchain_version/
│   └── agent_lc.py      # LangChain 重构版（对比框架与手写差异）
├── ui/app.py            # Streamlit 演示界面
├── data/sample_doc.md   # 示例语料（AI Agent 面试知识点手册）
├── tests/test_react.py  # 无 Key 可运行测试
├── docs/
│   ├── 使用说明.md       # 安装/换语料/升级向量检索/FAQ
│   └── 失败案例记录.md   # 失败与调优记录（面试必讲）
├── requirements.txt
└── .env.example
```

## 换成语料

在 `.env` 设置 `CORPUS_PATH` 指向你自己的文档（.md / .txt，UTF-8）：

```
CORPUS_PATH=data/sample_doc.md   # 改成你的路径，如 docs/我的课程笔记.md
```

建议选你**最熟悉**的一份文档（产品说明书/课程资料/技术笔记），面试才能讲深。

## 评测方法（面试关键）

1. 构造 20~30 个测试问题，手动标注每个问题应命中的段落
2. 跑检索，记录 top-3 命中率；逐步调 `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K`，一次只改一个变量
3. 人工核对回答准确率，记录失败类型（检索不准 / 幻觉 / 工具调用失败）
4. 结果写进 `docs/失败案例记录.md`

## Roadmap

- [x] 裸 ReAct 循环 + Function Calling
- [x] BM25 中文检索 RAG
- [x] LangChain 重构版（对比学习）
- [x] Streamlit 界面
- [ ] 接真实 Key 后跑完整评测，更新 README 效果示例
- [ ] 升级向量检索（通义 text-embedding）+ 重排序
