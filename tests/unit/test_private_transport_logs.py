import logging

from assistant.api.app import _protect_transport_logs
from assistant.nlu import itn


def test_debug_transport_logs_filtered_even_if_root_debug(monkeypatch, caplog):
    names = [
        "uvicorn.error",
        "websockets.client",
        "websockets.server",
        "httpcore2.connection",
        "httpcore2.http11",
        "openai._base_client",
    ]
    for name in names:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.DEBUG)
    monkeypatch.setattr(logging.getLogger("websockets"), "level", logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        _protect_transport_logs()
        for name in names:
            logging.getLogger(name).debug("secret-token-and-message")
            logging.getLogger(name).warning("transport-warning")
    assert "secret-token-and-message" not in caplog.text
    assert "transport-warning" in caplog.text


def test_do_not_weaken_stricter_log_levels(monkeypatch):
    logger = logging.getLogger("websockets.client")
    monkeypatch.setattr(logger, "level", logging.ERROR)
    _protect_transport_logs()
    assert logger.level == logging.ERROR


def test_itn_error_does_not_log_input_or_exception(monkeypatch, caplog):
    def broken(*args):
        raise ValueError("exception-with-private-data")

    monkeypatch.setattr(itn, "_get_alpha2digit", lambda: broken)
    with caplog.at_level(logging.DEBUG, logger="assistant.nlu.itn"):
        itn.numbers_to_digits("private-user-text")
    assert "alpha2digit skipped" in caplog.text
    assert "private-user-text" not in caplog.text
    assert "exception-with-private-data" not in caplog.text
