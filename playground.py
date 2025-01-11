from phi.agent import Agent
from phi.model.groq import Groq  # Assuming OpenAIModel exists in phi
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import phi.api
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv


import os
import phi
from phi.playground import Playground,serve_playground_app
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")


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

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground.app",reload=True)