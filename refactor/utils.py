import os
import logging
from typing import Awaitable, Callable, List, TypeVar

ROOT = os.path.join(os.path.basename(os.path.abspath(__file__)), "..")
SLACK_DATA_ROOT = os.path.join(ROOT, "slack_archive")

def get_logger():
    return logging.getLogger("uvicorn")

import asyncio

T = TypeVar("T")
R = TypeVar("R")

async def map_parallel(
    f: Callable[[T], Awaitable[R]],
    collection: List[T],
    *,
    semaphore: asyncio.Semaphore = asyncio.Semaphore(16),
) -> List[R]:
    async def worker(item: T) -> R:
        async with semaphore:
            return await f(item)

    tasks = [asyncio.create_task(worker(item)) for item in collection]
    return await asyncio.gather(*tasks)
