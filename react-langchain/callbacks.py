from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM was:***\n{prompts}")
        print("***********")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM stops running"""
        print(f"***LLM Response:***\n{response}")
        print("***********")