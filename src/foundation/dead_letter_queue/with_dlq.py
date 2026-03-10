"""`@with_dlq` decorator - inserts a DLQ instance into the decorated function."""

from functools import wraps
import inspect
from typing import Concatenate, ParamSpec, TypeVar, overload
from collections.abc import Callable, Awaitable

from foundation.dead_letter_queue.dlq import DLQ
from foundation.dead_letter_queue.dlq_backend import StubbedDLQBackend

P = ParamSpec("P")
T = TypeVar("T")

_DLQ_BACKEND = StubbedDLQBackend


@overload
def with_dlq(fn: Callable[Concatenate[DLQ, P], Awaitable[T]]) -> Callable[P, Awaitable[T]]: ...


@overload
def with_dlq(fn: Callable[Concatenate[DLQ, P], T]) -> Callable[P, T]: ...


def with_dlq(fn: Callable[..., T]) -> Callable[..., object]:
    """Injects a DLQ instance into the decorated function.

    The DLQ instance provides methods for registering failed image ingestions into a dead letter queue.

    The wrapped function *must* accept the DLQ as the first argument.

    Supports both sync and async functions.

    Example:
        ```
        @with_dlq
        def process_image(dlq: DLQ, image_id: str):
            try:
                do_something_with_image(image_id)
            except Exception as e:
                dlq.enqueue_failed_ingestion(image_id, metadata={"error": str(e)})
                raise
        ```
    """

    def create_dlq() -> DLQ:
        backend = _DLQ_BACKEND()
        return DLQ(backend)

    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
            dlq = create_dlq()

            return await fn(dlq, *args, **kwargs)

        wrapper_fn = async_wrapper

    else:

        @wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            dlq = create_dlq()

            return fn(dlq, *args, **kwargs)

        wrapper_fn = sync_wrapper

    setattr(wrapper_fn, "__wrapped_with_dlq", True)

    return wrapper_fn
