import threading
import ccxt
from typing import Dict, Tuple

class ExchangePool:
    """Singleton Connection Pool for CCXT exchange instances to prevent socket sızıntıları."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ExchangePool, cls).__new__(cls, *args, **kwargs)
                    cls._instance._exchanges = {}
        return cls._instance

    def get_exchange(self, exchange_id: str, enable_rate_limit: bool = True) -> ccxt.Exchange:
        ex_class = getattr(ccxt, exchange_id)
        
        # Check if the class is a mock (to prevent cache pollution during unit tests)
        is_mock = (
            hasattr(ex_class, "assert_called") 
            or hasattr(ex_class, "mock_add_spec") 
            or "mock" in type(ex_class).__name__.lower()
        )
        if is_mock:
            return ex_class({"enableRateLimit": enable_rate_limit})

        key = (exchange_id, enable_rate_limit)
        if key not in self._exchanges:
            with self._lock:
                if key not in self._exchanges:
                    ex = ex_class({"enableRateLimit": enable_rate_limit})
                    self._exchanges[key] = ex
        return self._exchanges[key]

    def clear(self):
        """Clear the cached exchange instances."""
        with self._lock:
            self._exchanges.clear()

_pool = ExchangePool()

def get_exchange(exchange_id: str, enable_rate_limit: bool = True) -> ccxt.Exchange:
    """Retrieve a cached singleton instance of a CCXT exchange."""
    return _pool.get_exchange(exchange_id, enable_rate_limit)
