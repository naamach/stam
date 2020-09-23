from configparser import ConfigParser
import logging
import os


def get_config(config_file="config.ini"):
    config = ConfigParser(inline_comment_prefixes=';')
    config.read(config_file)
    return config


def init_log(filename="log.log", config_file="config.ini"):
    config = get_config(config_file)
    log_path = config.get('LOG', 'PATH')  # log file path
    console_log_level = config.get('LOG', 'CONSOLE_LEVEL')  # logging level
    file_log_level = config.get('LOG', 'FILE_LEVEL')  # logging level

    # create log folder
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # console handler
    h = logging.StreamHandler()
    h.setLevel(logging.getLevelName(console_log_level))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(formatter)
    log.addHandler(h)

    # log file handler
    h = logging.FileHandler(log_path + filename + ".log", "w", encoding=None, delay=True)
    h.setLevel(logging.getLevelName(file_log_level))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(formatter)
    log.addHandler(h)

    return log


def close_log(log):
    handlers = list(log.handlers)
    for h in handlers:
        log.removeHandler(h)
        h.flush()
        h.close()
    return
