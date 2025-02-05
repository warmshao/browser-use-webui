from behave import when, then
from browser_use import Agent
import asyncio

@then('I should see the company logo for {company}')
def step_impl(context, company):
    context.task = f"verify that the {company} logo is visible on the page"
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

@then('I should see images of the {landmark}')
def step_impl(context, landmark):
    context.task = f"verify that images of the {landmark} are visible in the search results"
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

@when('I switch to dark mode')
def step_impl(context):
    context.task = "switch the website to dark mode"
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=True
        )
        return await agent.run(max_steps=5)
    
    context.history = context.loop.run_until_complete(run_agent())

@then('the background should be dark')
def step_impl(context):
    context.task = "verify that the page background is dark"
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

@then('text should be light colored')
def step_impl(context):
    context.task = "verify that the text color is light against the dark background"
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

@then('I should see the OpenAI logo')
def step_impl(context):
    context.task = "verify that the OpenAI logo is visible on the page"
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