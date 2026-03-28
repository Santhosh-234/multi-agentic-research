import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool  
import yfinance as yf


os.environ["OPENAI_API_KEY"] = "your_api_key" 
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"

ddg_search = DuckDuckGoSearchRun()


@tool
def search_tool(query: str) -> str:
    """Search the web for the latest financial news, product launches, and company updates."""
    return ddg_search.run(query)


@tool
def stock_price_tool(ticker: str) -> str:
    """Fetch the last 5 days of closing prices for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return f"Recent price data for {ticker}:\n{hist['Close'].to_string()}"
    except Exception as e:
        return f"Error fetching data: {e}"


researcher = Agent(
    role='Senior Market Researcher',
    goal='Gather the most recent and impactful news regarding {company}',
    backstory='You are a veteran Wall Street researcher. You excel at finding the signal in the noise and identifying news that actually moves markets.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

quant_analyst = Agent(
    role='Quantitative Analyst',
    goal='Analyze the recent price action and volume trends for {company}',
    backstory='You are a data-driven quant. You care about numbers, trends, and moving averages. You ignore rumors and focus on the tape.',
    verbose=True,
    allow_delegation=False,
    tools=[stock_price_tool]
)

cio = Agent(
    role='Chief Investment Officer',
    goal='Synthesize research and quant data to make a definitive Buy/Hold/Sell recommendation for {company}',
    backstory='You manage a billion-dollar portfolio. You need concise, actionable insights. You weigh risks heavily and demand logical justification for every trade.',
    verbose=True,
    allow_delegation=False
)


task_research = Task(
    description='Search the web for the latest news, product launches, or executive changes regarding {company}. Summarize the top 3 most critical points.',
    expected_output='A bulleted summary of the 3 most important recent news items for the company.',
    agent=researcher
)

task_quant = Task(
    description='Fetch the recent stock price history for {company} (Ticker: {ticker}) and identify if the trend is bullish, bearish, or flat.',
    expected_output='A brief quantitative analysis of the 5-day price trend.',
    agent=quant_analyst
)

task_recommendation = Task(
    description='Review the news summary and the quantitative analysis. Write a 2-paragraph investment thesis for {company} ending with a bolded BUY, HOLD, or SELL recommendation.',
    expected_output='A professional investment thesis with a clear final recommendation.',
    agent=cio
)


financial_crew = Crew(
    agents=[researcher, quant_analyst, cio],
    tasks=[task_research, task_quant, task_recommendation],
    process=Process.sequential, 
    verbose=True
)

if __name__ == "__main__":
    print("--- BOOTING FINANCIAL MULTI-AGENT SYSTEM ---")
    
    inputs = {
        'company': 'NVIDIA',
        'ticker': 'NVDA'
    }
    
    result = financial_crew.kickoff(inputs=inputs)
    
    print("\n================================================")
    print("FINAL CIO INVESTMENT REPORT:")
    print("================================================")
    print(result)