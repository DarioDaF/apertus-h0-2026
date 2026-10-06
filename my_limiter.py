'''
    Description: Implementation of a limiter class that can define constraint for sync and async calls
    Author: DarioDaF
'''

import math
from threading import BoundedSemaphore, Lock
from time import monotonic, sleep
from contextlib import contextmanager, asynccontextmanager

class MyLimiter:
    def __init__(self, *, min_delay_between: float, max_concurrent: int):
        self.min_delay_between = min_delay_between
        self.max_concurrent = max_concurrent
        self._slots = BoundedSemaphore(self.max_concurrent)
        self._last_lock = Lock()
        self._last_start = -math.inf
    @contextmanager
    def acquire(self):
        with self._slots:
            with self._last_lock:
                sleep(max(0.0, self.min_delay_between - (monotonic() - self._last_start)))
                self._last_start = monotonic()
            yield
    @asynccontextmanager
    async def aacquire(self):
        raise NotImplementedError('Async limiters are not implemented')
        yield

class MyNoopLimiter:
    @contextmanager
    def acquire(self):
        yield
    @asynccontextmanager
    async def aacquire(self):
        yield
myNoopLimiter = MyNoopLimiter()
