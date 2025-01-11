from phi.agent import Agent
from phi.model.groq import Groq  # Assuming OpenAIModel exists in phi
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Use OpenAIModel or equivalent
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Define Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Use OpenAIModel or equivalent
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,
)

# Multi AI Agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVIDIA", stream=True
)
