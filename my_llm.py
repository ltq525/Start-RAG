# my_llm.py
import os
from typing import Any, Iterator, Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException


class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        # 检查provider是否为我们想处理的'mimo'
        if provider == "mimo":
            print("正在使用自定义的 MiMo Provider")
            self.provider = "mimo"
            
            # 解析 MiMo 的凭证
            self.api_key = api_key or os.getenv("MIMO_API_KEY") or os.getenv("LLM_API_KEY")
            self.base_url = base_url or os.getenv("MIMO_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.mimo-v2.com/v1"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("MiMo API key not found. Please set MIMO_API_KEY or LLM_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("MIMO_MODEL_ID") or os.getenv("LLM_MODEL_ID") or "mimo-v2-pro"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        elif provider == "ollama":
            print("正在使用 Ollama Provider")
            # 这里可以添加针对 Ollama 的特殊处理逻辑
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)

        else:
            # 如果不是 mimo, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
    
    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        自动检测LLM提供商
        """
        # 1. 检查特定提供商的环境变量 (最高优先级)
        if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
        if os.getenv("OPENAI_API_KEY"): return "openai"
        if os.getenv("ZHIPU_API_KEY"): return "zhipu"
        if os.getenv("MIMO_API_KEY"): return "mimo"
        # ... 其他服务商的环境变量检查

        # 获取通用的环境变量
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 2. 根据 base_url 判断
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
            if "open.bigmodel.cn" in base_url_lower: return "zhipu"
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower: return "ollama"
                if ":8000" in base_url_lower: return "vllm"
                return "local" # 其他本地端口
            if "mimo" in base_url_lower: return "mimo"

        # 3. 根据 API 密钥格式辅助判断
        if actual_api_key:
            if actual_api_key.startswith("ms-"): return "modelscope"
            # ... 其他密钥格式判断

        # 4. 默认返回 'auto'，使用通用配置
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """根据provider解析API密钥和base_url"""
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url

        elif self.provider == "mimo":
            resolved_api_key = api_key or os.getenv("MIMO_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("MIMO_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.mimo-v2.com/v1"
            return resolved_api_key, resolved_base_url

        return super()._resolve_credentials(api_key, base_url)

    def _get_default_model(self) -> str:
        """获取默认模型。"""
        if self.provider == "mimo":
            return os.getenv("MIMO_MODEL_ID") or os.getenv("LLM_MODEL_ID") or "mimo-v2.5-pro"
        return super()._get_default_model()

    @staticmethod
    def _get_stream_delta_content(chunk: Any) -> str:
        """兼容 choices 为空的 OpenAI-compatible 流式事件。"""
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""

        content = getattr(delta, "content", None)
        if content:
            return content

        reasoning_content = getattr(delta, "reasoning_content", None)
        return reasoning_content or ""

    @staticmethod
    def _get_message_content(message: Any) -> str:
        """兼容正文出现在 content 或 reasoning_content 的响应。"""
        content = getattr(message, "content", None)
        if content:
            return content

        reasoning_content = getattr(message, "reasoning_content", None)
        return reasoning_content or ""

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型并返回流式响应。

        部分 OpenAI-compatible 服务会在流式响应中发送 choices=[] 的
        usage/metadata 事件，父类直接访问 choices[0] 会触发 IndexError。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            print("✅ 大语言模型开始返回流式响应")
            for chunk in response:
                content = self._get_stream_delta_content(chunk)
                if content:
                    yield content

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """非流式调用LLM。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            return self._get_message_content(response.choices[0].message)
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """流式调用LLM。"""
        temperature = kwargs.get("temperature")
        yield from self.think(messages, temperature)



    pass
