from behave import when, then
from browser_use import Agent
import asyncio

@when('I type "{text}" in the search box')
def step_impl(context, text):
    context.task = f"type '{text}' in the search box"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I click the search button')
def step_impl(context):
    context.task = "click the search button"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I click the first link')
def step_impl(context):
    context.task = "click the first link in the search results"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I scroll to the {position} of the page')
def step_impl(context, position):
    context.task = f"scroll to the {position} of the page"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I scroll to the top')
def step_impl(context):
    context.task = "scroll to the top of the page"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@then('the page title should contain "{text}"')
def step_impl(context, text):
    context.task = f"verify that the page title contains '{text}'"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())
    assert context.history.errors() == []

@then('I should see a page element: {element}')
def step_impl(context, element):
    context.task = f"verify that the {element} is visible"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=True
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())
    assert context.history.errors() == []

@then('I should see the header')
def step_impl(context):
    context.task = "verify that the page header is visible"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=True
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())
    assert context.history.errors() == []

@then('I should see the footer')
def step_impl(context):
    context.task = "verify that the page footer is visible"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=True
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())
    assert context.history.errors() == [] 