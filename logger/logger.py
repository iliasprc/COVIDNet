import logging
import logging.config
import os
import sys
import time

from utils.util import make_dirs_if_not_present


class Timer:
    """
     A simple timer class for tracking time.
     """
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y-%m-%d-%H:%M:%S"
    DEFAULT_TIME_FORMAT = ["%03dms", "%02ds", "%02dm", "%02dh"]

    def __init__(self):
        """
        Initializes the timer.
        """

        self.start = time.time() * 1000

    def get_current(self) -> str:
        """
        Returns the current time.

        Returns:
            str: The current time in milliseconds.
        """

        return self.get_time(self.start)

    def reset(self):
        """
        Resets the timer.
        """
        self.start = time.time() * 1000

    def get_time_since_start(self, time_format: list=None) -> str:
        """
        Get the time since the timer started.

        Args:
            time_format (list, optional): Time format to be returned.

        Returns:
            str: The time elapsed since the timer started.
        """
        return self.get_time(self.start, time_format)

    def get_time(self, start: float = None, end: float = None, time_format: list = None) -> str:
        """
        Calculates the elapsed time between start and end. If no start time is provided, it returns the current time.

        Args:
            start (float, optional): Start time in milliseconds.
            end (float, optional): End time in milliseconds.
            time_format (list, optional): Time format to be returned.

        Returns:
            str: The elapsed time or the current time if start is None.
        """
        if start is None:
            if time_format is None:
                time_format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(time_format)

        if end is None:
            end = time.time() * 1000
        time_elapsed = end - start

        if time_format is None:
            time_format = self.DEFAULT_TIME_FORMAT

        s, ms = divmod(time_elapsed, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        items = [ms, s, m, h]
        assert len(items) == len(time_format), "Format length should be same as items"

        time_str = ""
        for idx, item in enumerate(items):
            if item != 0:
                time_str = time_format[idx] % item + " " + time_str

        # Means no more time is left
        if len(time_str) == 0:
            time_str = "0ms"

        return time_str.strip()


class Logger:
    """
    A logger class for logging messages.
    """
    def __init__(self, path: str, log_level: str=None, name: str=None):
        """
        Initializes the logger.

        Args:
            path (str): The path where logs will be saved.
            log_level (str, optional): The level of logging.
            name (str, optional): The name of the logger.
        """
        self.logger = None
        self.timer = Timer()

        self.log_filename = "train_"
        self.log_filename += self.timer.get_time()
        self.log_filename += ".log"

        self.log_folder = os.path.join(path, 'logs/')
        make_dirs_if_not_present(self.log_folder)

        self.log_filename = os.path.join(self.log_folder, self.log_filename)

        logging.captureWarnings(True)

        if not name:
            name = __name__

        self.logger = logging.getLogger(name)

        # Set level
        if log_level is None:
            level = 'INFO'
        else:
            level = log_level
        self.logger.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d-%H:%M:%S",
        )

        # Add handlers
        file_hdl = logging.FileHandler(self.log_filename)
        file_hdl.setFormatter(formatter)
        self.logger.addHandler(file_hdl)
        # logging.getLogger('py.warnings').addHandler(file_hdl)
        cons_hdl = logging.StreamHandler(sys.stdout)
        cons_hdl.setFormatter(formatter)
        self.logger.addHandler(cons_hdl)

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger object.

        Returns:
            logging.Logger: The logger object.
        """
        return self.logger