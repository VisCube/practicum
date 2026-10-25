from assistant import Assistant


def main() -> None:
    assistant = Assistant()

    while True:
        user_text = input("> ").strip()
        response, tool_calls = assistant.handle_message(user_text)

        for call in tool_calls:
            print(f"[{call.name}] {call.arguments} -> {call.result}")

        print(response)


if __name__ == "__main__":
    main()
