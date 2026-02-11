from loguru import logger


def log_debug(message: str, module: str | None = None, **kwargs):
    _get_logger(module).debug(message, **kwargs)


def log_warning(message: str, module: str | None = None, **kwargs):
    _get_logger(module).warning(message, **kwargs)


def log_error(message: str, module: str | None = None, **kwargs):
    _get_logger(module).error(message, **kwargs)


def _get_logger(module: str | None = None):
    return logger.bind(module=module) if module else logger
