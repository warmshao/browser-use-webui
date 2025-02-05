from behave import given, when, then
from browser_use import Agent
import asyncio

@given('I am on "{url}"')
def step_impl(context, url):
    # Handle both direct URLs and special cases
    if url.lower() == "google.com" or url.lower() == "the google homepage":
        context.task = "navigate to google.com"
    else:
        context.task = f"navigate to {url}"
    
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=10)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I navigate to "{url}"')
def step_impl(context, url):
    context.task = f"go to {url}"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=10)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I go back in browser history')
def step_impl(context):
    context.task = "click the browser's back button"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@when('I go forward in browser history')
def step_impl(context):
    context.task = "click the browser's forward button"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=False
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@then('I should see "{text}" in the page')
def step_impl(context, text):
    context.task = f"check if the text '{text}' is visible on the page"
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

@then('I should be on "{url}"')
def step_impl(context, url):
    context.task = f"verify that we are on {url}"
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