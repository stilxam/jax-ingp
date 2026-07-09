import logging
import os


def setup_logger(out_dir: str, name: str = "train") -> logging.Logger:
    """Logs to both stdout and `<out_dir>/train.log`. Both `StreamHandler`
    and `FileHandler` flush on every emitted record, so log lines land
    immediately (unlike a bare `print()` piped to a file, which is
    block-buffered and can sit invisible until the process exits)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(out_dir, "train.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
