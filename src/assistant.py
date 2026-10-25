from json import loads

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall as FunctionToolCall,
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionToolMessageParam as ToolMessage,
    ChatCompletionUserMessageParam as UserMessage
)

from src.config import FM_API_KEY, FM_BASE_URL, LLM_MODEL, SYSTEM_PROMPT
from src.tools.__init__ import handle_tool_call, TOOL_SCHEMAS


class ToolCall:
    def __init__(self, name: str, arguments: dict, result: str):
        self.name = name
        self.arguments = arguments
        self.result = result


class Assistant:
    def __init__(self):
        self.client = OpenAI(api_key=FM_API_KEY, base_url=FM_BASE_URL)
        self.messages: list[Message] = [
            SystemMessage(role="system", content=SYSTEM_PROMPT)
        ]

    def handle_message(self, user_input: str) -> tuple[str, list[ToolCall]]:
        self.messages.append(UserMessage(role="user", content=user_input))
        tool_calls: list[ToolCall] = []

        while True:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or "", tool_calls

            for tool_call in message.tool_calls:
                if not isinstance(tool_call, FunctionToolCall):
                    continue

                result = handle_tool_call(tool_call)
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        arguments=loads(tool_call.function.arguments),
                        result=result,
                    )
                )

                self.messages.append(
                    ToolMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=result,
                    )
                )
