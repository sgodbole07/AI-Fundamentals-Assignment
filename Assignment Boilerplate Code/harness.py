"""Agent Harness — YOU IMPLEMENT THIS.

The harness orchestrates the agentic loop:
  user input → LLM → tool call → validate → execute → feed result back → repeat
"""

from core.llm_client import MockLLMClient
from core.tool_registry import ToolRegistry
from core.validator import Validator
from core.logger import StepLogger


class AgentHarness:
    """The agent harness — ties LLM, tools, validator, and logger into the agentic loop."""

    def __init__(self, llm_client: MockLLMClient, tool_registry: ToolRegistry,
                 validator: Validator, logger: StepLogger, max_steps: int = 5):
        """Initialize the harness.

        Args:
            llm_client: The (mocked) LLM client
            tool_registry: Registry of available tools
            validator: Validates tool calls before execution
            logger: Logs each step
            max_steps: Maximum loop iterations before forced stop
        """
        self._llm = llm_client
        self._registry = tool_registry
        self._validator = validator
        self._logger = logger
        self._max_steps = max_steps

    def run(self, user_input: str) -> dict:
        """Run the agent loop for a given user input.

        Args:
            user_input: The user's natural language request

        Returns:
            {
                "final_answer": str,       # The agent's final text response
                "steps_taken": int,        # How many loop iterations ran
                "stop_reason": str,        # Why the loop stopped:
                                           #   "final_answer" - LLM gave text response
                                           #   "max_steps" - hit step limit
                                           #   "validation_error" - tool call invalid
                                           #   "approval_denied" - user rejected dangerous tool
                "log": list[dict]          # Full step log from logger
            }
        """
        messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]

        for step in range(1, self._max_steps + 1):

            response = self._llm.chat(messages)

            self._logger.log_step(
                step,
                "llm_response",
                response
            )

            if response.get("type") == "text":
                final_answer = response.get("content", "")

                self._logger.log_step(
                    step,
                    "final_answer",
                    {
                        "content": final_answer,
                        "stop_reason": "final_answer"
                    }
                )

                return {
                    "final_answer": final_answer,
                    "steps_taken": step,
                    "stop_reason": "final_answer",
                    "log": self._logger.get_log()
                }

            validation = self._validator.validate_tool_call(response)

            self._logger.log_step(
                step,
                "validation",
                validation
            )

            if not validation.get("valid", False):
                error_message = validation.get(
                    "error",
                    "Invalid tool call"
                )

                self._logger.log_step(
                    step,
                    "final_answer",
                    {
                        "content": error_message,
                        "stop_reason": "validation_error"
                    }
                )

                return {
                    "final_answer": error_message,
                    "steps_taken": step,
                    "stop_reason": "validation_error",
                    "log": self._logger.get_log()
                }

            tool_name = response["tool_name"]
            arguments = response.get("arguments", {})


            self._logger.log_step(
                step,
                "tool_call",
                {
                    "tool_name": tool_name,
                    "arguments": arguments
                }
            )

            tool_result = self._registry.execute_tool(
                tool_name,
                arguments
            )

            self._logger.log_step(
                step,
                "tool_result",
                {
                    "result": tool_result
                }
            )

            messages.append(
                {
                    "role": "tool_result",
                    "content": tool_result
                }
            )

        return {
            "final_answer": "",
            "steps_taken": self._max_steps,
            "stop_reason": "max_steps",
            "log": self._logger.get_log()
        }
