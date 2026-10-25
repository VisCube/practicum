from json import dumps, loads
from pathlib import Path

from openai.types.chat import (
    ChatCompletionFunctionToolParam as FunctionTool,
    ChatCompletionMessageFunctionToolCall as FunctionToolCall
)
from openai.types.shared_params import FunctionDefinition

from src.tools.search import list_knowledge_files, search_knowledge_base
from src.tools.crm import create_ticket

CATEGORIES_PATH = Path(__file__).parent.parent.parent / "classifier" / "categories.json"
CATEGORIES: list[str] = loads(CATEGORIES_PATH.read_text(encoding="utf-8"))

TOOL_SCHEMAS = [
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
                        "description": "Фамилия, имя, отчество"
                    },
                    "address": {
                        "type": "string",
                        "description": "Адрес: улица, дом, квартира"
                    },
                    "details": {
                        "type": "string",
                        "description": "Подробное описание обращения"
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "Наиболее подходящая категория обращения из классификатора"
                    },
                },
                "required": ["full_name", "address", "details", "category"]
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
    ),
    FunctionTool(
        type="function",
        function=FunctionDefinition(
            name="search_knowledge_base",
            description="Прочитать полный текст файла базы знаний по его точному имени",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": f"Точное имя файла из списка доступных файлов: {list_knowledge_files()}"
                    }
                },
                "required": ["filename"]
            }
        )
    )
]


def transfer_to_operator(reason: str) -> str:
    print(f"[Перевод на оператора: {reason}]")
    return "Диалог переведён на оператора."


def handle_tool_call(tool_call: FunctionToolCall) -> str:
    function = TOOL_HANDLERS[tool_call.function.name]
    arguments = loads(tool_call.function.arguments)

    result = function(**arguments)
    return dumps(result, ensure_ascii=False)


TOOL_HANDLERS = {
    "create_ticket": create_ticket,
    "transfer_to_operator": transfer_to_operator,
    "search_knowledge_base": search_knowledge_base
}
