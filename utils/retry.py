import time
import random
import logging
from typing import Callable, TypeVar, Tuple, Type

T = TypeVar("T")
log = logging.getLogger("retry")

def with_retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """Decorator to retry a function with exponential backoff and jitter."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Do not retry on non-transient ExchangeError if subclassed from ccxt.ExchangeError
                    # but check if we can import ccxt safely inside
                    try:
                        import ccxt
                        if isinstance(e, ccxt.ExchangeError) and not isinstance(e, ccxt.DDoSProtection):
                            # It's a config/arg error, raising immediately
                            raise e
                    except ImportError:
                        pass
                    
                    if attempt == retries - 1:
                        log.error(f"Function {func.__name__} failed permanently after {retries} attempts: {e}")
                        raise e
                    
                    log.warning(
                        f"Error in {func.__name__} (attempt {attempt + 1}/{retries}): {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                    if jitter:
                        delay += random.uniform(0, 0.1 * delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator
