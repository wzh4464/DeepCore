import logging
import functools
import traceback


def log_exception(logger=None):
    """
    装饰器：自动捕获并记录函数中的异常。
    用法：@log_exception(logger)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                nonlocal logger
                if logger is None:
                    logger = logging.getLogger(func.__module__)
                logger.error(f"Exception in {func.__name__}: {e}")
                logger.error(traceback.format_exc())
                raise
        return wrapper
    return decorator


class ExceptionHandler:
    """
    统一异常处理类，可扩展自定义处理逻辑。
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def handle(self, e, context=None):
        msg = f"Exception: {e}"
        if context:
            msg = f"[{context}] {msg}"
        self.logger.error(msg)
        self.logger.error(traceback.format_exc())
        # 可扩展：如写入数据库、上报监控等
        # raise  # 如需中断流程可取消注释 
