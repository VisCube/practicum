from dataclasses import asdict
from json import dumps, loads

from openai.types.chat import (
    ChatCompletionFunctionToolParam as FunctionTool,
    ChatCompletionMessageFunctionToolCall as FunctionToolCall
)
from openai.types.shared_params import FunctionDefinition

from src.tools.search import search_knowledge_base
from src.tools.telephony import transfer_to_operator
from src.tools.classifier import CATEGORIES_NAMES, get_category
from src.tools.crm.residents import Resident, get_resident_by_phone
from src.tools.crm.tickets import (
    create_ticket,
    get_ticket,
    get_tickets_by_address,
)

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
            name="get_resident_by_phone",
            description="Получить жильца по номеру телефона",
            parameters={
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "Номер телефона жильца"
                    }
                },
                "required": ["phone"]
            }
        )
    ),
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="get_ticket",
            description="Получить заявку по её номеру",
            parameters={
                "type": "object",
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "Номер заявки"
                    }
                },
                "required": ["number"]
            }
        )
    ),
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="get_tickets_by_address",
            description="Получить заявки по адресу.",
            parameters={
                "type": "object",
                "properties": {
                    "street": {
                        "type": "string",
                        "description": "Название улицы"
                    },
                    "house": {
                        "type": "string",
                        "description": "Номер дома"
                    },
                    "apartment": {
                        "type": "integer",
                        "description": "Номер квартиры, если категория относится к персональным"
                    }
                },
                "required": ["street", "house"]
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
                    "phone": {
                        "type": "string",
                        "description": "Номер телефона звонящего"
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES_NAMES,
                        "description": "Наиболее подходящая категория обращения из классификатора"
                    },
                    "description": {
                        "type": "string",
                        "description": "Краткая суммаризация обращения"
                    }
                },
                "required": ["category", "phone", "description"],
                "additionalProperties": {"type": ["string", "integer"]}
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


def handle_tool_call(tool_call: FunctionToolCall) -> str:
    function = TOOL_HANDLERS[tool_call.function.name]
    arguments = loads(tool_call.function.arguments)

    result = function(**arguments)
    if isinstance(result, Resident):
        result = {**asdict(result), "phone": result.phone.number}
    return dumps(result, ensure_ascii=False, default=vars)


TOOL_HANDLERS = {
    "get_resident_by_phone": get_resident_by_phone,
    "get_ticket": get_ticket,
    "get_tickets_by_address": get_tickets_by_address,
    "get_category": get_category,
    "create_ticket": create_ticket,
    "transfer_to_operator": transfer_to_operator,
    "search_knowledge_base": search_knowledge_base
}
