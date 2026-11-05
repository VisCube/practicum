from json import dumps, loads

from openai.types.chat import (
    ChatCompletionFunctionToolParam as FunctionTool,
    ChatCompletionMessageFunctionToolCall as FunctionToolCall
)
from openai.types.shared_params import FunctionDefinition

from src.tools.search import search_knowledge_base
from src.tools.crm import CATEGORIES_NAMES, create_ticket, get_category

TOOL_SCHEMAS = [
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="get_category",
            description="Получить категорию и её JSON Schema",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES_NAMES
                    }
                },
                "required": ["category"]
            }
        )
    ),
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="create_ticket",
            description="Создать заявку в CRM",
            parameters={
                "type": "object",
                "properties": {
                    "full_name": {
                        "type": "string",
                        "description": "Фамилия, имя, отчество звонящего полностью"
                    },
                    "address": {
                        "type": "string",
                        "description": "Улица и дом звонящего"
                    },
                    "details": {
                        "type": "string",
                        "description": "Описание обращения"
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES_NAMES,
                        "description": "Наиболее подходящая категория обращения из классификатора"
                    },
                },
                "required": ["full_name", "address", "details", "category"],
                "additionalProperties": {"type": ["string", "number"]}
            }
        )
    ),
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="search_knowledge_base",
            description="Найти ответ на вопрос в базе знаний",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Вопрос звонящего"
                    }
                },
                "required": ["query"]
            }
        )
    ),
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="transfer_to_operator",
            description="Перевести диалог на живого оператора",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Причина перевода на оператора"
                    },
                },
                "required": ["reason"]
            }
        )
    )
]


def transfer_to_operator(reason: str) -> str:
    return reason


def handle_tool_call(tool_call: FunctionToolCall) -> str:
    function = TOOL_HANDLERS[tool_call.function.name]
    arguments = loads(tool_call.function.arguments)

    result = function(**arguments)
    return dumps(result, ensure_ascii=False)


TOOL_HANDLERS = {
    "get_category": get_category,
    "create_ticket": create_ticket,
    "transfer_to_operator": transfer_to_operator,
    "search_knowledge_base": search_knowledge_base
}
