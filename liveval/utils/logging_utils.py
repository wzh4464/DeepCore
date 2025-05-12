import os
import sys
import logging
from datetime import datetime
import torch

class GpuPrefixFilter(logging.Filter):
    """
    日志Filter，为每条日志加上[GPU-x]或[CPU]前缀。
    """
    def __init__(self, gpu_id=None):
        super().__init__()
        self.gpu_prefix = self.detect_gpu_prefix(gpu_id)

    def detect_gpu_prefix(self, gpu_id):
        if torch.cuda.is_available():
            # 获取当前进程实际使用的物理GPU编号（取第一个可用）
            try:
                # 优先用CUDA_VISIBLE_DEVICES映射到物理编号
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if cuda_visible:
                    # 取第一个可用的物理编号
                    visible_list = [int(x) for x in cuda_visible.split(",") if x.strip().isdigit()]
                    if visible_list:
                        # 取当前进程的默认GPU
                        # current = torch.cuda.current_device()
                        # 物理编号
                        return f"[GPU-{visible_list[gpu_id]}]"
                # 否则直接用物理编号
                # current = torch.cuda.current_device()
                return f"[GPU-{gpu_id}]"
            except Exception:
                return "[GPU-?]"
        else:
            return "[CPU]"

    def filter(self, record):
        record.gpu_prefix = self.gpu_prefix
        return True

def setup_logging(log_dir="logs", log_level=logging.INFO, log_name=None, gpu_id=None):
    """
    设置统一的日志配置，支持文件和控制台输出，并为每条日志加上[GPU-x]或[CPU]前缀。

    Args:
        log_dir (str): 日志文件夹路径
        log_level: 日志级别
        log_name (str): 日志文件名（可选）
    Returns:
        logger: 配置好的logger实例
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_name:
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(gpu_prefix)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # 给所有handler加上GPU前缀Filter
    gpu_filter = GpuPrefixFilter(gpu_id)
    for handler in logging.getLogger().handlers:
        handler.addFilter(gpu_filter)
    logger = logging.getLogger()
    logger.info(f"日志初始化完成，日志文件: {log_file}")
    return logger

def get_logger(name=None):
    """
    获取指定名称的logger，便于模块内统一调用。
    """
    return logging.getLogger(name) 