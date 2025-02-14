import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

async def execute_with_interaction(agent_state, agent_func, *args, **kwargs):
    """
    Execute an agent function with support for pausing, resuming, and context updates.
    
    Args:
        agent_state: The AgentState instance managing execution state
        agent_func: The main agent function to execute
        *args, **kwargs: Arguments to pass to the agent function
    """
    try:
        # Start execution
        logger.info("Starting execution with interaction support")
        
        # Execute the agent function with pause checks
        async def wrapped_agent_func(*args, **kwargs):
            while True:
                # Check for stop request first
                if agent_state.is_stop_requested():
                    logger.info("Stop requested, terminating execution")
                    return None

                # Check and handle pause
                if await agent_state.is_paused():
                    logger.info("Execution paused, waiting for resume...")
                    while await agent_state.is_paused():
                        if agent_state.is_stop_requested():
                            logger.info("Stop requested while paused")
                            return None
                        await asyncio.sleep(0.1)
                    logger.info("Execution resumed")

                try:
                    # Get any pending tasks
                    if await agent_state.has_pending_tasks():
                        task = await agent_state.get_next_task()
                        logger.info(f"Processing task: {task}")
                        await agent_state.update_context({"current_task": task})

                    # Execute one step of the agent function
                    result = await agent_func(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.error(f"Error during execution step: {str(e)}")
                    await agent_state.update_context({"last_error": str(e)})
                    raise

        # Run the wrapped function
        return await wrapped_agent_func(*args, **kwargs)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        await agent_state.update_context({"last_error": str(e)})
        raise 