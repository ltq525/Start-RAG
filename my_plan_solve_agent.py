# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


import ast
import re
from typing import Dict, List, Optional

from hello_agents import Config, Message, PlanAndSolveAgent
from hello_agents.core.llm import HelloAgentsLLM


class MyPlanner:
    """规划器：负责把复杂问题拆成按顺序执行的步骤。"""

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_PLANNER_PROMPT
        self.system_prompt = system_prompt

    def plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        messages = self._build_messages(prompt)

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        plan = self._parse_plan(response_text)
        if not plan:
            print(f"❌ 无法解析有效计划，原始响应: {response_text}")
        return plan

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_plan(self, response_text: str) -> List[str]:
        candidates = self._extract_list_candidates(response_text)
        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                return [str(step).strip() for step in parsed if str(step).strip()]

        numbered_steps = []
        for line in response_text.splitlines():
            match = re.match(r"^\s*(?:[-*]|\d+[.)、])\s*(.+?)\s*$", line)
            if match:
                numbered_steps.append(match.group(1))
        return numbered_steps

    def _extract_list_candidates(self, text: str) -> List[str]:
        candidates = []

        fenced_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
        candidates.extend(block.strip() for block in fenced_blocks)

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1].strip())

        candidates.append(text.strip())
        return [candidate for candidate in candidates if candidate]


class MyExecutor:
    """执行器：按照计划逐步调用 LLM，并保留每一步结果。"""

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_EXECUTOR_PROMPT
        self.system_prompt = system_prompt
        self.history = ""

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        self.history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for index, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {index}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=self.history if self.history else "无",
                current_step=step,
            )
            messages = self._build_messages(prompt)
            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            self.history += f"步骤 {index}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {index} 已完成，结果: {final_answer}")

        return final_answer

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """自定义 Plan-and-Solve Agent：先规划，再按步骤执行。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)

        custom_prompts = custom_prompts or {}
        self.planner = MyPlanner(
            llm_client=self.llm,
            prompt_template=custom_prompts.get("planner"),
            system_prompt=self.system_prompt,
        )
        self.executor = MyExecutor(
            llm_client=self.llm,
            prompt_template=custom_prompts.get("executor"),
            system_prompt=self.system_prompt,
        )
        print(f"✅ {name} 初始化完成，Plan-and-Solve 模式已启用")

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
