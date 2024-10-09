
# Platform-specific optimizations
if IS_WINDOWS:
    import win32api
    import win32process
    
    def set_process_priority(priority: int):
        handle = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(handle, priority)

elif IS_POSIX:
    import resource

    def set_process_priority(priority: int):
        try:
            os.nice(priority)
        except PermissionError:
            print("Warning: Unable to set process priority. Running with default priority.")
def memoize(func: Callable) -> Callable: # caching decorator
    return lru_cache(maxsize=None)(func)

def log(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
#-------------------------------#########TYPING################-------------------------------#
class AtomType(Enum):
    FUNCTION = auto() # FIRST CLASS FUNCTIONS
    VALUE = auto()
    CLASS = auto() # CLASSES ARE FUNCTIONS, BUT CAN HAVE A CLASS POLYMORPH
    MODULE = auto() # SimpleNameSpace()(s) are MODULE (~MODULE IS A SNS)




#-------------------------------###############################-------------------------------#
#-------------------------------###############################-------------------------------#
#-------------------------------###############################-------------------------------#


#-------------------------------######DECORATORS###############-------------------------------#
