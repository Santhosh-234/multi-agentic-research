import os
import json
import uuid
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool
import chromadb 


os.environ["OPENAI_API_KEY"] = "your_api_key" 
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"


print("Initializing ChromaDB Vector Vault...")
chroma_client = chromadb.PersistentClient(path="./vector_vault")
reports_collection = chroma_client.get_or_create_collection(name="financial_reports")


class InvestmentReport(BaseModel):
    ticker: str = Field(description="The stock ticker symbol")
    recommendation: str = Field(description="Strictly 'BUY', 'HOLD', or 'SELL'")
    risk_level: str = Field(description="Strictly 'Low', 'Medium', or 'High'")
    executive_summary: str = Field(description="A 2-paragraph investment thesis")

class ResearchRequest(BaseModel):
    company_name: str
    ticker: str


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


app = FastAPI(title="Agentic Equity Research API")

@app.post("/api/v1/research", response_model=InvestmentReport)
async def generate_research_report(request: ResearchRequest):
    print(f"\n🚀 Initiating Agentic Research for: {request.company_name} ({request.ticker})")
    
    researcher = Agent(
        role='Senior Market Researcher',
        goal=f'Gather the most recent and impactful news regarding {request.company_name}',
        backstory='You are a veteran Wall Street researcher. Find the signal in the noise.',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )

    quant_analyst = Agent(
        role='Quantitative Analyst',
        goal=f'Analyze the recent price action and volume trends for {request.ticker}',
        backstory='You are a data-driven quant. You care about numbers, trends, and moving averages.',
        verbose=True,
        allow_delegation=False,
        tools=[stock_price_tool]
    )

    cio = Agent(
        role='Chief Investment Officer',
        goal=f'Synthesize research and quant data to make a definitive Buy/Hold/Sell recommendation for {request.company_name}',
        backstory='You manage a billion-dollar portfolio. You demand logical justification for every trade.',
        verbose=True,
        allow_delegation=False
    )

    task_research = Task(
        description=f'Search the web for the latest news regarding {request.company_name}. Summarize top 3 points.',
        expected_output='Bulleted summary of the 3 most important recent news items.',
        agent=researcher
    )

    task_quant = Task(
        description=f'Fetch recent stock price history for {request.ticker} and identify the trend.',
        expected_output='Brief quantitative analysis of the 5-day price trend.',
        agent=quant_analyst
    )

    task_recommendation = Task(
        description=f'Review the news and quant analysis. Generate a final investment thesis for {request.ticker}. YOU MUST OUTPUT ONLY VALID JSON. Do not include markdown, backticks, or conversational text. Use exactly these keys: "ticker", "recommendation", "risk_level", "executive_summary".',
        expected_output='A raw JSON object.',
        agent=cio
    )

    financial_crew = Crew(
        agents=[researcher, quant_analyst, cio],
        tasks=[task_research, task_quant, task_recommendation],
        process=Process.sequential,
        verbose=True
    )

    try:
        crew_output = financial_crew.kickoff()
        raw_string = crew_output.raw.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(raw_string)
        
        
        report_id = str(uuid.uuid4())
        reports_collection.add(
            documents=[parsed_json["executive_summary"]],
            metadatas=[{
                "ticker": parsed_json["ticker"], 
                "recommendation": parsed_json["recommendation"], 
                "risk_level": parsed_json.get("risk_level", "Unknown")
            }],
            ids=[report_id]
        )
        print(f"✅ Report for {parsed_json['ticker']} vectorized and archived to ChromaDB Vault.")
        
        return parsed_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")


@app.get("/api/v1/search_vault")
async def search_historical_reports(query: str, limit: int = 2):
    try:
        results = reports_collection.query(
            query_texts=[query],
            n_results=limit
        )
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)