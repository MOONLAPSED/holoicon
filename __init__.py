
# Setup custom logging format for enhanced error messages and debugging
class customFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setupLogger(name: str, level: int, datefmt: str, handlers: list):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    for handler in handlers:
        if not isinstance(handler, logging.Handler):
            raise ValueError(f"Invalid handler provided: {handler}")
        handler.setLevel(level)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)

    return logger

def logArgs():
    parser = argparse.ArgumentParser(description="Logger Configuration")
    parser.add_argument('--log-level', type=str, default='DEBUG', choices=logging._nameToLevel.keys(), help='Set logging level')
    parser.add_argument('--log-file', type=str, help='Set log file path')
    parser.add_argument('--log-format', type=str, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)', help='Set log format')
    parser.add_argument('--log-datefmt', type=str, default='%Y-%m-%d %H:%M:%S', help='Set date format')
    parser.add_argument('--log-name', type=str, default=__name__, help='Set logger name')
    return parser.parse_args()
def parseLargs():
    args = parse_args()
    log_level = logging._nameToLevel.get(args.log_level.upper(), logging.DEBUG)

    handlers = [logging.FileHandler(args.log_file)] if args.log_file else [logging.StreamHandler()]

    logger = setup_logger(name=args.log_name, level=log_level, datefmt=args.log_datefmt, handlers=handlers)
    logger.info("Logger setup complete.")

def load_modules():
    try:
        mixins = []
        for path in pathlib.Path(__file__).parent.glob("*.py"):
            if path.name.startswith("_"):
                continue
            module_name = path.stem
            spec = spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module {module_name}")
            module = module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            mixins.append(module)
        return mixins
    except Exception as e:
        Logger.error(f"Error importing internal modules: {e}")
        sys.exit(1)

# Import the internal modules
mixins = load_modules()

if mixins:
    __all__ = [mixin.__name__ for mixin in mixins]
else:
    __all__ = []

# Handle platform-specific dynamic linking logic
if IS_POSIX:
    try:
        from ctypes import cdll
        Logger.info("POSIX system detected.")
    except ImportError:
        Logger.error("Error loading POSIX dynamic linking libraries.")
else:
    try:
        from ctypes import windll
        Logger.info("Windows system detected.")
    except ImportError:
        Logger.error("Error loading Windows dynamic linking libraries.")

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import ast

""" hacked namespace uses `__all__` as a whitelist of symbols which are executable source code.
Non-whitelisted modules or runtime constituents are treated as 'data' which we call associative 
'articles' within the knowledge base, loaded at runtime."""

class KnowledgeBase:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.globals = SimpleNamespace()
        self.globals.__all__ = []
        self.initialize()

    def initialize(self):
        self._import_py_modules(self.base_dir)
        self._load_articles(self.base_dir)

    def _import_py_modules(self, directory):
        for path in directory.rglob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                module_name = path.stem
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                setattr(self.globals, module_name, module)
                self.globals.__all__.append(module_name)
            except Exception as e:
                print(f"Error importing module {module_name}: {e}")

    def _load_articles(self, directory):
        for suffix in ['*.md', '*.txt']:
            for path in directory.rglob(suffix):
                try:
                    article_name = path.stem
                    content = path.read_text()
                    article = SimpleNamespace(
                        content=content,
                        path=str(path)
                    )
                    setattr(self.globals, article_name, article)
                except Exception as e:
                    print(f"Error loading article from {path}: {e}")

    def execute_query(self, query):
        try:
            parsed = ast.parse(query, mode='eval')
            result = eval(compile(parsed, '<string>', 'eval'), {'kb': self.globals})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def commit_changes(self):
        # TODO: Implement logic to write changes back to the file system
        pass

def initialize_kb(base_dir):
    return KnowledgeBase(base_dir)