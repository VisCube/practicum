"""Read-only local CLI. Output contains personal data; restrict shell access."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный просмотр журнала звонков")
    parser.add_argument("--db", required=True, help="путь к SQLite")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--limit", type=int, default=50)
    show = sub.add_parser("show")
    show.add_argument("call_id")
    args = parser.parse_args()
    if args.command == "list" and not 1 <= args.limit <= 1000:
        parser.error("--limit должен быть от 1 до 1000")
    path = Path(args.db).resolve(strict=True)
    with sqlite3.connect("file:" + quote(str(path)) + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        if args.command == "list":
            data = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM calls ORDER BY started_at DESC LIMIT ?", (args.limit,)
                )
            ]
        else:
            call = db.execute(
                "SELECT * FROM calls WHERE journal_id=? OR call_id=? "
                "ORDER BY started_at DESC LIMIT 1",
                (args.call_id, args.call_id),
            ).fetchone()
            if call is None:
                parser.exit(1, "Звонок не найден\n")
            events = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM call_events WHERE journal_id=? ORDER BY id",
                    (call["journal_id"],),
                )
            ]
            data = {"call": dict(call), "events": events}
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
