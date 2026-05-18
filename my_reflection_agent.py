DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


from typing import Any, Dict, List, Optional

from hello_agents import Config, HelloAgentsLLM, Message, ReflectionAgent


class ReflectionMemory:
    """简单短期记忆，用来保存执行结果和反思反馈。"""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str) -> None:
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_last_execution(self) -> str:
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

    def get_trajectory(self) -> str:
        trajectory = []
        for record in self.records:
            if record["type"] == "execution":
                trajectory.append(f"--- 执行结果 ---\n{record['content']}")
            elif record["type"] == "reflection":
                trajectory.append(f"--- 反思反馈 ---\n{record['content']}")
        return "\n\n".join(trajectory)


class MyReflectionAgent(ReflectionAgent):
    """
    自定义 Reflection Agent。

    工作流程:
    1. 根据任务生成初始回答
    2. 反思当前回答的问题和改进空间
    3. 根据反馈优化回答
    4. 重复迭代，直到无需改进或达到最大迭代次数
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config, max_iterations)
        self.max_iterations = max_iterations
        self.memory = ReflectionMemory()
        self.prompts = DEFAULT_PROMPTS.copy()
        if custom_prompts:
            self.prompts.update(custom_prompts)
        print(f"✅ {name} 初始化完成，最大反思轮数: {max_iterations}")

    def run(self, input_text: str, **kwargs) -> str:
        """执行任务，并通过反思迭代优化最终结果。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        self.memory = ReflectionMemory()

        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration + 1}/{self.max_iterations} 轮迭代 ---")

            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result,
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            if self._is_good_enough(feedback):
                print("\n✅ 反思认为结果已无需改进，任务完成。")
                break

            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                content=last_result,
                feedback=feedback,
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")
        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.invoke(messages, **kwargs) or ""

    def _is_good_enough(self, feedback: str) -> bool:
        normalized = feedback.lower()
        return "无需改进" in feedback or "no need for improvement" in normalized
