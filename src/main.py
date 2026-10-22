from assistant import Assistant


def main() -> None:
    assistant = Assistant()

    while True:
        user_text = input("> ").strip()
        reply = assistant.handle_message(user_text)
        print(reply)


if __name__ == "__main__":
    main()
