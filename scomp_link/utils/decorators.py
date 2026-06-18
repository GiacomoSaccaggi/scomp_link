# -*- coding: utf-8 -*-
"""
██████╗ ███████╗ █████╗  █████╗ ██████╗  █████╗ ████████╗ █████╗ ██████╗ ██████╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔════╝
██║  ██║█████╗  ██║  ╚═╝██║  ██║██████╔╝███████║   ██║   ██║  ██║██████╔╝╚█████╗
██║  ██║██╔══╝  ██║  ██╗██║  ██║██╔══██╗██╔══██║   ██║   ██║  ██║██╔══██╗ ╚═══██╗
██████╔╝███████╗╚█████╔╝╚█████╔╝██║  ██║██║  ██║   ██║   ╚█████╔╝██║  ██║██████╔╝
╚═════╝ ╚══════╝ ╚════╝  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚════╝ ╚═╝  ╚═╝╚═════╝

General-purpose decorators for scomp-link.
"""

import time
import traceback
import functools
import warnings
from typing import Any, Callable, Optional

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)



def timer(func: Callable) -> Callable:
    """
    Print execution time of a function.

    Example:
        @timer
        def train_model(X, y):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed < 60:
            logger.info(f"⏱️  {func.__name__} completed in {elapsed:.2f}s")
        else:
            minutes, seconds = divmod(elapsed, 60)
            logger.info(f"⏱️  {func.__name__} completed in {int(minutes)}m {seconds:.1f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Retry a function on failure.

    Example:
        @retry(max_attempts=3, delay=2.0)
        def fetch_data(url):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    logger.info(f"⚠️  {func.__name__} failed (attempt {attempt}/{max_attempts}): {e}")
                    time.sleep(delay)
        return wrapper
    return decorator


def log_call(func: Callable) -> Callable:
    """
    Log function calls with arguments and return value.

    Example:
        @log_call
        def process(df, threshold=0.5):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a)[:50] for a in args]
        kwargs_repr = [f"{k}={repr(v)[:30]}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.info(f"📞 {func.__name__}({signature})")
        result = func(*args, **kwargs)
        logger.info(f"   ↳ returned {type(result).__name__}")
        return result
    return wrapper


def memory_usage(func: Callable) -> Callable:
    """
    Print peak memory usage of a function (requires tracemalloc).

    Example:
        @memory_usage
        def load_big_dataset():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"🧠 {func.__name__} — peak memory: {peak / 1024 / 1024:.1f} MB")
        return result
    return wrapper


def cache(func: Callable) -> Callable:
    """
    Simple in-memory cache (memoization) for function results.

    Example:
        @cache
        def expensive_computation(n):
            ...
    """
    memo = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in memo:
            memo[key] = func(*args, **kwargs)
        else:
            logger.info(f"💾 {func.__name__} — cache hit")
        return memo[key]

    wrapper.cache_clear = memo.clear
    return wrapper


def deprecated(message: str = ""):
    """
    Mark a function as deprecated. Emits a warning on each call.

    Example:
        @deprecated("Use new_function() instead")
        def old_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"⚠️  {func.__name__} is deprecated. {message}".strip()
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def suppress_exceptions(default: Any = None, log: bool = True):
    """
    Catch all exceptions and return a default value instead of crashing.

    Example:
        @suppress_exceptions(default=pd.DataFrame())
        def risky_transform(df):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log:
                    logger.info(f"❌ {func.__name__} failed: {e}")
                return default
        return wrapper
    return decorator


def validate_args(**validators):
    """
    Validate function arguments before execution.

    Example:
        @validate_args(df=lambda x: len(x) > 0, threshold=lambda x: 0 <= x <= 1)
        def process(df, threshold=0.5):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for '{param_name}' in {func.__name__}: got {repr(value)[:50]}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def run_once(func: Callable) -> Callable:
    """
    Ensure a function is only executed once. Subsequent calls return the cached result.

    Example:
        @run_once
        def initialize_model():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper._called:
            wrapper._result = func(*args, **kwargs)
            wrapper._called = True
        return wrapper._result

    wrapper._called = False
    wrapper._result = None
    return wrapper
