import logging
import os
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__ from calling module)
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler (shows INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (shows DEBUG and above) - detailed logs
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_log_filename(script_name):
    """
    Generate a timestamped log filename
    
    Args:
        script_name: Name of the script (e.g., 'data_acquisition_python')
    
    Returns:
        Log file path with timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"logs/{script_name}_{timestamp}.log"
