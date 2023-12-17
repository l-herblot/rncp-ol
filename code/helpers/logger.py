import inspect
import logging
import sys
from os import path
from pathlib import Path


class CustomLogger(logging.Logger):
    """This class overrides logging.Logger to add some logging features :
    - Logging the exception details (when applicable)
    - Logging the "trigger" (caller function + path)

    The recommended use for the different levels of logging is :
    - debug() : Detailed information, typically only of interest to a developer trying to diagnose
    a problem.
    - info() : Confirmation that things are working as expected.
    - warning() : An indication that something unexpected happened, or that a problem might occur
    in the near future (e.g. ‘disk space low’). The software is still working as expected.
    - error() : Due to a more serious problem, the software has not been able to perform some
    function.
    - critical() : A serious error, indicating that the program itself may be unable to continue
    running. We chose to use this only if a "vital" feature is failing or about to fail
    Note that critical() will send an email as its default behavior."""

    def __init__(
        self,
        name,
        level=logging.NOTSET,
        logging_path=(str(Path(__file__).absolute().parent.parent / "logs") + path.sep),
    ):
        """The class' constructor.
        Requires a name and, optionally, the minimum level required to log entries, the default
        email addresses and the path to store logging files"""
        super(CustomLogger, self).__init__(name, level)
        self.logging_path = logging_path
        self.propagate = False
        # The logging format mask to pass to the super class
        self.format_mask = ""
        # The root directory for the project so that the logged path to the calling script stops
        # there (e.g. "/usr/co2/projects/projet/api/whatever.py" would become
        # "api > whatever.py" if root_dir="projet")
        self.root_dir = ""

    def _get_trigger(self):
        """Return a string composed of the path to the caller file and the caller function in the
        format : path > to > the > file.py::caller_function"""
        stack = inspect.stack()
        stack_info = stack[3] if len(stack) > 3 else stack[2]
        caller_path = stack_info[1]

        caller_path = caller_path.split("\\" if "\\" in caller_path else "/")

        if isinstance(caller_path, list):
            opath = caller_path
            caller_path = []
            for name in reversed(opath):
                if name == self.root_dir:
                    break
                caller_path.append(name)

            caller_path = ">".join(reversed(caller_path))

        return f"{caller_path}::{stack_info[3]}"

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stack_level=1,
    ):
        """This is the main function. It overrides the default logging.Logger._log() function
        - level is the logging level
        - msg is the message to log
        - args, exc_info, extra, stack_info, stack-level are optional args to be passed to super._log()
        """
        trigger = self._get_trigger()
        _, exception_value, _ = sys.exc_info()
        exception_msg = (
            f" (Exception : {exception_value})" if exception_value is not None else ""
        )

        # The default _log function is called after all our processing
        super(CustomLogger, self)._log(
            level,
            trigger + " | " + msg + exception_msg,
            args,
            exc_info,
            extra,
            stack_info,
            stack_level,
        )


def get_logger(
    name,
    root_dir="",
    format_mask="%(asctime)s | %(levelname)s@%(name)s | %(message)s",
):
    """Instantiate a new CustomLogger object and initialize it with the given arguments and default values/behaviors
    - name is the logger's name (aka. id)
    - root_dir is the base directory name for the project. The logged path will stop when encountering
    this name. e.g. base directory is set to "projet" and the function whatevs (located in
    /usr/co2/projects/projet/api/whatever.py) calls the logger. The logged "trigger"
    will be api > whatever.py::whatevs
    - format_mask is the output format mask for the default _log function. The default one should be good enough
    """
    logger = logging.getLogger(name)
    logger.format_mask = format_mask
    logger.root_dir = root_dir

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if format_mask:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(format_mask, "%Y-%m-%d %H:%M:%S")
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def init_logging(level=logging.INFO):
    """Initialize the logging library
    level is the logging level under which no regular logging will occur.
    CustomLogger isn't impacted by this argument.
    It should be one of :
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    """
    global logger

    if logger is not None:
        return

    logging.setLoggerClass(CustomLogger)

    logging.basicConfig(level=level)

    # Optimizations for the logging process
    # Deactivate collection for threading information
    logging.logThreads = False
    # Deactivate collection for current process ID (os.getpid())
    logging.logProcesses = False
    # Deactivate collection for current process name when using multiprocessing
    logging.logMultiprocessing = False
    # Deactivate collection for current asyncio.Task name
    logging.logAsyncioTasks = False

    logger = get_logger(f"co2_{__name__}", "projet")


# Initialize the logger object
logger = None
init_logging()
