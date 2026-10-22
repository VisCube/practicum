from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall as FunctionToolCall,
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionToolMessageParam as ToolMessage,
    ChatCompletionUserMessageParam as UserMessage
)

from src.config import FM_API_KEY, FM_BASE_URL, LLM_MODEL, SYSTEM_PROMPT
from src.tools.__init__ import TOOL_SCHEMAS, handle_tool_call


class Assistant:
    def __init__(self):
        self.client = OpenAI(api_key=FM_API_KEY, base_url=FM_BASE_URL)
        self.messages: list[Message] = [
            SystemMessage(role="system", content=SYSTEM_PROMPT)
        ]

    def handle_message(self, user_input: str) -> str:
        self.messages.append(UserMessage(role="user", content=user_input))

        while True:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                if not isinstance(tool_call, FunctionToolCall):
                    continue

                self.messages.append(
                    ToolMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=handle_tool_call(tool_call),
                    )
                )
