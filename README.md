# Hello Agents

一个基于 Python 的智能体学习与实验项目，围绕 OpenAI-compatible LLM 接口实现了多种 Agent 范式、工具系统、记忆模块、RAG 能力，以及一个 Gradio PDF 文档问答示例应用。

项目既包含 `hello_agents/` 里的框架化代码，也包含根目录下的 `my_*.py`、`test_*.py` 示例脚本，用于演示如何扩展 LLM Provider、Agent、工具链和文档问答流程。

## 功能概览

- 统一 LLM 客户端：`HelloAgentsLLM` 支持 OpenAI、DeepSeek、通义千问 DashScope、ModelScope、Kimi、智谱、Ollama、vLLM、本地 OpenAI-compatible 服务等。
- 自定义 MiMo Provider：`my_llm.py` 扩展了 MiMo 配置识别，并兼容部分流式响应事件中 `choices=[]` 的情况。
- 多种 Agent 范式：基础对话 Agent、ReAct Agent、Reflection Agent、Plan-and-Solve Agent。
- 工具系统：包含工具基类、工具注册表、计算器、搜索、工具链和异步工具执行。
- 记忆系统：支持工作记忆、情景记忆、语义记忆、感知记忆，并可结合 Qdrant、Neo4j 等后端。
- RAG 能力：支持文档解析、向量化、知识库检索和基于 LLM 的问答。
- PDF 学习助手：`Q&A_Assistant.py` 提供 Gradio Web UI，可上传 PDF、提问、记录笔记、回顾学习历史并生成学习报告。

## 项目结构

```text
.
├── hello_agents/
│   ├── agents/          # Simple/ReAct/Reflection/Plan-and-Solve Agent
│   ├── core/            # LLM、Agent 基类、配置、消息与异常
│   ├── memory/          # 记忆管理、Embedding、RAG、存储后端
│   ├── tools/           # 工具基类、注册表、内置工具、工具链
│   ├── utils/           # 日志、序列化与辅助函数
│   └── version.py
├── my_*.py              # 自定义 LLM、Agent、工具示例
├── test_*.py            # 演示脚本，多数需要真实 LLM 配置
├── Q&A_Assistant.py     # Gradio PDF 文档问答助手
├── requirement.txt      # 依赖列表
├── .env.example         # 环境变量模板，不包含真实密钥
└── .gitignore           # 忽略密钥、本地缓存和运行产物
```

## 快速开始

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

复制环境变量模板，并填写自己的模型、搜索和数据库配置：

```bash
cp .env.example .env
```

最小可用配置通常只需要填写：

```dotenv
LLM_MODEL_ID=your-model
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
```

如果使用搜索工具，可额外配置 `TAVILY_API_KEY` 或 `SERPAPI_API_KEY`。如果使用 RAG/记忆能力，可按需配置 `QDRANT_*`、`NEO4J_*` 和 `EMBED_*`。

## 运行示例

基础 LLM 调用：

```bash
python my_main.py
```

基础 Agent 与计算器工具演示：

```bash
python test_simple_agent.py
```

Reflection Agent 演示：

```bash
python test_reflection_agent.py
```

Plan-and-Solve Agent 演示：

```bash
python test_plan_solve_agent.py
```

启动 PDF 文档问答助手：

```bash
python "Q&A_Assistant.py"
```

默认会读取 `.env` 中的 `GRADIO_SERVER_NAME` 和 `GRADIO_SERVER_PORT`。示例配置会监听 `127.0.0.1:7860`。

## 代码示例

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool
from my_simple_agent import MySimpleAgent

load_dotenv()

llm = HelloAgentsLLM()
tool_registry = ToolRegistry()
tool_registry.register_tool(CalculatorTool())

agent = MySimpleAgent(
    name="计算助手",
    llm=llm,
    system_prompt="你是一个可以使用工具的智能助手。",
    tool_registry=tool_registry,
)

print(agent.run("请帮我计算 15 * 8 + 32"))
```

## 环境变量说明

`.env` 是本地私密配置文件，已经被 `.gitignore` 忽略。请不要提交真实 API Key、数据库密码、云服务实例 ID 或本地凭据。

常用配置：

- `LLM_MODEL_ID`：模型名称。
- `LLM_API_KEY`：OpenAI-compatible 服务密钥。
- `LLM_BASE_URL`：OpenAI-compatible API 地址。
- `TAVILY_API_KEY` / `SERPAPI_API_KEY`：搜索工具密钥。
- `QDRANT_URL` / `QDRANT_API_KEY`：向量数据库连接。
- `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`：图数据库连接。
- `EMBED_MODEL_TYPE` / `EMBED_MODEL_NAME` / `EMBED_API_KEY`：Embedding 模型配置。
- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT`：Gradio 服务监听地址。

## 敏感文件与本地产物

本仓库已添加忽略规则，默认过滤：

- `.env`、`.env.*` 等真实环境配置。
- 私钥、证书、凭据 JSON 等敏感文件。
- `__pycache__/`、`*.pyc`、`.DS_Store` 等本地生成文件。
- `.venv/`、构建目录、测试缓存、日志文件。
- `knowledge_base/`、`memory_data/`、`learning_report_*.json` 等运行时数据。
- 本地 Qdrant、Neo4j、SQLite、模型缓存等可能较大或含私密数据的文件。

如果后续又有生成文件被误提交，需要从索引中移除一次，之后 `.gitignore` 才会生效：

```bash
git rm -r --cached path/to/__pycache__
git rm --cached path/to/.DS_Store
```

这些命令只会从 Git 索引中移除文件，不会删除你本地磁盘上的文件。

## 开发提示

- 根目录下的 `test_*.py` 更接近演示脚本，不是完全离线的单元测试；多数脚本需要有效的 `.env` LLM 配置。
- RAG 和 PDF 助手会创建本地知识库、记忆数据和学习报告，相关目录已被忽略。
- 若要接入新的 OpenAI-compatible 服务，优先通过 `LLM_*` 环境变量配置；需要特殊兼容时可参考 `my_llm.py`。
- 若要新增工具，继承 `hello_agents.tools.base.Tool` 并注册到 `ToolRegistry` 即可。
