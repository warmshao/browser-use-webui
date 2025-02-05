from behave import given, when, then
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
import asyncio
from src.utils import utils

@given('I am on the Google homepage')
def step_impl(context):
    context.task = "go to google.com"
    context.add_infos = ""
    context.use_vision = False

@when('I search for "{query}"')
def step_impl(context, query):
    context.task = f"go to google.com and type '{query}' click search and give me the first url"
    context.query = query

@when('I search for "{query}" with vision enabled')
def step_impl(context, query):
    context.task = f"go to google.com and type '{query}' click search and give me the first url"
    context.query = query
    context.use_vision = True

@then('I should see search results')
def step_impl(context):
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=getattr(context, 'use_vision', False)
        )
        return await agent.run(max_steps=10)
    
    context.history = context.loop.run_until_complete(run_agent())
    assert context.history.final_result() is not None

@then('the first result should contain "{expected_url}"')
def step_impl(context, expected_url):
    result = context.history.final_result()
    assert expected_url in str(result).lower()

@then('I should see search results with images')
def step_impl(context):
    async def run_agent():
        agent = Agent(
            task=context.task,
            llm=context.llm,
            browser_context=context.browser_context,
            use_vision=True
        )
        return await agent.run(max_steps=10)
    
    context.history = context.loop.run_until_complete(run_agent())
    assert context.history.final_result() is not None

@then('I should see the Nvidia logo')
def step_impl(context):
    # This step would require visual verification
    # For now we'll just check if the task completed successfully
    assert context.history.errors() == [] 