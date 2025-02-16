import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
import sys

sys.path.append(".")

async def test_deep_research():
    from src.utils.deep_research import deep_research
    from src.utils import utils
    
    task = "write a report about DeepSeek-R1, get its pdf"
    llm = utils.get_llm_model(
        provider="gemini",
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        temperature=1.0,
        api_key=os.getenv("GOOGLE_API_KEY", "")
    )
    
    report_content, report_file_path = await deep_research(task=task, llm=llm, agent_state=None, 
                                                           max_search_iterations=1, 
                                                           max_query_num=3,
                                                           use_own_browser=False)
    


if __name__ == "__main__":
    asyncio.run(test_deep_research())