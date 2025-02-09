import asyncio
import logging

logger = logging.getLogger(__name__)

class AgentState:
    _instance = None

    def __init__(self):
        if not hasattr(self, '_stop_requested'):
            self._stop_requested = asyncio.Event()
            self.last_valid_state = None
            self._current_task = None
            self._task_queue = asyncio.Queue()
            self._context = {}
            self._paused = asyncio.Event()
            self._pause_condition = asyncio.Condition()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentState, cls).__new__(cls)
        return cls._instance

    def request_stop(self):
        self._stop_requested.set()

    def clear_stop(self):
        self._stop_requested.clear()
        self.last_valid_state = None

    def is_stop_requested(self):
        return self._stop_requested.is_set()

    async def pause_execution(self):
        """Pause the current task execution"""
        async with self._pause_condition:
            self._paused.set()
            logger.info("Execution paused")

    async def resume_execution(self):
        """Resume the paused task execution"""
        async with self._pause_condition:
            self._paused.clear()
            self._pause_condition.notify_all()
            logger.info("Execution resumed")

    async def wait_if_paused(self):
        """Wait if execution is paused"""
        if self._paused.is_set():
            async with self._pause_condition:
                await self._pause_condition.wait()

    async def is_paused(self):
        """Check if execution is paused"""
        return self._paused.is_set()

    async def update_context(self, context: dict):
        """Update execution context"""
        self._context.update(context)
        logger.info(f"Context updated: {context}")

    async def get_context(self) -> dict:
        """Get current execution context"""
        return self._context

    def set_last_valid_state(self, state):
        self.last_valid_state = state

    def get_last_valid_state(self):
        return self.last_valid_state

    async def add_task(self, task):
        """Add a new task to the queue"""
        await self._task_queue.put(task)

    async def get_next_task(self):
        """Get the next task from the queue"""
        return await self._task_queue.get()

    async def has_pending_tasks(self):
        """Check if there are pending tasks"""
        return not self._task_queue.empty()