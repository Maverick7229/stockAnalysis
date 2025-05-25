import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import re

# LLM
from langchain_groq import ChatGroq

# Tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

# Agent and Agent Executor
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- 1. LLM Configuration ---
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="gemma2-9b-it"
)

# --- 2. Improved Tool Definitions ---

# Enhanced YFinance Tool with better error handling
class ImprovedYFinanceTool:
    @staticmethod
    def get_stock_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info with error handling
            try:
                info = stock.info
            except Exception as e:
                print(f"Warning: Could not fetch info for {ticker}: {str(e)}")
                info = {}
            
            # Get historical data for better analysis
            try:
                hist = stock.history(period="2y")  # 2 years of data
                if hist.empty:
                    raise Exception("No historical data available")
            except Exception as e:
                print(f"Warning: Could not fetch historical data for {ticker}: {str(e)}")
                hist = pd.DataFrame()
            
            # Extract key metrics with fallbacks
            current_price = None
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            elif 'currentPrice' in info:
                current_price = info['currentPrice']
            elif 'previousClose' in info:
                current_price = info['previousClose']
            
            # Calculate 52-week high/low from historical data if available
            if not hist.empty:
                fifty_two_week_high = hist['High'].max()
                fifty_two_week_low = hist['Low'].min()
                
                # Find the date of highest and lowest prices
                high_date = hist[hist['High'] == fifty_two_week_high].index[0].strftime('%Y-%m-%d')
                low_date = hist[hist['Low'] == fifty_two_week_low].index[0].strftime('%Y-%m-%d')
                
                # Calculate recent trend (last 30 days vs previous 30 days)
                recent_avg = hist['Close'].tail(30).mean()
                previous_avg = hist['Close'].tail(60).head(30).mean()
                trend_direction = "UP" if recent_avg > previous_avg else "DOWN"
                trend_percentage = ((recent_avg - previous_avg) / previous_avg) * 100
            else:
                fifty_two_week_high = info.get("fiftyTwoWeekHigh", "N/A")
                fifty_two_week_low = info.get("fiftyTwoWeekLow", "N/A")
                high_date = "N/A"
                low_date = "N/A"
                trend_direction = "N/A"
                trend_percentage = 0
            
            # Get recent news with error handling
            try:
                news = stock.news[:3] if hasattr(stock, 'news') and stock.news else []
                news_titles = [item.get('title', 'No title') for item in news]
            except:
                news_titles = ["News data unavailable"]
            
            # Market cap calculation
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A' and market_cap:
                market_cap_crore = market_cap / 10000000  # Convert to crores
            else:
                market_cap_crore = 'N/A'
            
            return {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName", "Unknown")),
                "current_price_inr": round(current_price, 2) if current_price else "N/A",
                "52_week_high_inr": round(fifty_two_week_high, 2) if isinstance(fifty_two_week_high, (int, float)) else fifty_two_week_high,
                "52_week_low_inr": round(fifty_two_week_low, 2) if isinstance(fifty_two_week_low, (int, float)) else fifty_two_week_low,
                "high_date": high_date,
                "low_date": low_date,
                "market_cap_crore": round(market_cap_crore, 2) if isinstance(market_cap_crore, (int, float)) else market_cap_crore,
                "recent_trend": f"{trend_direction} ({trend_percentage:.1f}%)",
                "volume": info.get("volume", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "recent_news": news_titles,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A")
            }
            
        except Exception as e:
            return {
                "ticker": ticker,
                "error": f"Error fetching data for {ticker}: {str(e)}",
                "company_name": "N/A",
                "current_price_inr": "N/A"
            }

@tool("YahooFinanceStockDataTool")
def yfinance_stock_data_tool(ticker: str) -> dict:
    """
    Fetches comprehensive stock information for Indian stocks using Yahoo Finance.
    Input should be a stock ticker symbol (e.g., 'ZEEL.NS' for NSE or 'ZEEL.BO' for BSE).
    Provides current price, 52-week high/low, market cap, trends, and recent news.
    """
    return ImprovedYFinanceTool.get_stock_info(ticker)

# Enhanced Web Search Tool
web_search_tool = DuckDuckGoSearchRun()

# --- 3. Predefined list of known Indian penny stocks for fallback ---
KNOWN_PENNY_STOCKS = [
    # Telecom/Media
    "ZEEL.NS", "SITI.NS", "RCOM.NS",
    # Infrastructure/Real Estate  
    "JETAIRWAYS.NS", "SUZLON.NS", "YESBANK.NS",
    # Textile/Manufacturing
    "RPOWER.NS", "JPASSOCIAT.NS", "RNAVAL.NS",
    # Others
    "PCJEWELLER.NS", "GTLINFRA.NS", "BHARTIINFRA.NS",
    # BSE equivalents
    "ZEEL.BO", "SUZLON.BO", "YESBANK.BO"
]

# --- 4. Enhanced Agent Definitions ---

# Agent 1: Web Search Agent with better prompting
web_search_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert Web Researcher for Indian Financial Markets. Current date: May 2025. "
        "DEFINITION: Indian penny stocks are typically stocks trading below ₹50 or with market cap below ₹500 crore. "
        "Your task is to find specific Indian penny stock ticker symbols that have experienced significant volatility. "
        "Search for stocks that have fallen from highs and might be recovering. "
        "Focus on finding actual NSE (.NS) or BSE (.BO) ticker symbols. "
        "Look for recent financial news about penny stocks, market crashes, recoveries, etc. "
        "Always try to extract specific company names and their ticker symbols from search results."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

web_search_agent_runnable = create_openai_tools_agent(
    llm=llm,
    tools=[web_search_tool],
    prompt=web_search_agent_prompt
)

web_search_executor = AgentExecutor(
    agent=web_search_agent_runnable,
    tools=[web_search_tool],
    verbose=True,
)

# Agent 2: Enhanced Financial Analyst Agent
financial_analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a specialist Indian Stock Market Analyst. Current date: May 2025. "
        "Analyze Indian penny stocks to find 'fallen angels' - stocks that hit highs, then lows, now showing recovery. "
        "Use the YahooFinanceStockDataTool to get detailed data for each ticker. "
        "Look for: 1) Significant gap between 52-week high and low, 2) Current price recovery from lows, "
        "3) Positive recent trends, 4) Reasonable market cap for penny stock category. "
        "Present findings in structured format with clear reasoning for each selection."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

financial_analyst_runnable = create_openai_tools_agent(
    llm=llm,
    tools=[yfinance_stock_data_tool],
    prompt=financial_analyst_prompt
)

financial_analyst_executor = AgentExecutor(
    agent=financial_analyst_runnable,
    tools=[yfinance_stock_data_tool],
    verbose=True,
)

# --- 5. Helper function to extract tickers from text ---
def extract_tickers_from_text(text):
    """Extract ticker symbols from text"""
    pattern = r'\b[A-Z][A-Z0-9]*\.(NS|BO)\b'
    tickers = re.findall(pattern, text)
    return [f"{ticker[0]}.{ticker[1]}" for ticker in tickers]

# --- 6. Enhanced Orchestration ---
def run_stock_analysis():
    print("## Starting Enhanced Indian Penny Stock Analysis ##")
    
    # Step 1: Web search with improved prompting
    task1_input = (
        "Search for Indian penny stocks that have experienced significant volatility in 2023-2024. "
        "Look for specific company names and ticker symbols (.NS or .BO format). "
        "Focus on stocks that: 1) Were trading at higher prices in 2023-2024, "
        "2) Experienced significant drops, 3) Might be showing signs of recovery in early 2025. "
        "Examples of sectors to look at: telecom (Zee Entertainment, Vodafone Idea), "
        "infrastructure (Suzlon, Reliance Power), banking (Yes Bank), aviation (Jet Airways). "
        "Try to find at least 8-10 specific ticker symbols with their current status."
    )
    
    print(f"\n--- Step 1: Enhanced Web Search for Penny Stocks ---")
    
    try:
        response_step1 = web_search_executor.invoke({
            "input": task1_input,
            "chat_history": []
        })
        
        web_output = response_step1['output']
        print(f"Web Search Results:\n{web_output}")
        
        # Extract tickers from web search results
        found_tickers = extract_tickers_from_text(web_output)
        
        # If no tickers found, use our predefined list
        if not found_tickers:
            print("\nNo tickers found in web search. Using predefined penny stock list...")
            tickers_to_analyze = KNOWN_PENNY_STOCKS[:10]  # Use first 10
        else:
            tickers_to_analyze = found_tickers[:10]  # Limit to 10
            
        print(f"\nTickers to analyze: {tickers_to_analyze}")
        
    except Exception as e:
        print(f"Web search failed: {str(e)}")
        print("Using predefined penny stock list...")
        tickers_to_analyze = KNOWN_PENNY_STOCKS[:10]
    
    # Step 2: Financial analysis with actual tickers
    task2_input = (
        f"Analyze these Indian penny stock tickers: {', '.join(tickers_to_analyze)}. "
        "For each ticker, use YahooFinanceStockDataTool to fetch data and determine if it fits the criteria: "
        "1) Notable high in last 1-2 years, 2) Significant drop afterward, 3) Signs of recent recovery. "
        "Focus on stocks with: Current price < ₹50, Market cap < ₹500 crore, "
        "52-week high significantly higher than current price, recent upward trend. "
        "Select the top 5 stocks that best fit the 'fallen angel with recovery potential' pattern. "
        "For each selected stock provide: Company name, ticker, current price, "
        "52-week high/low, market cap, recent trend, and investment thesis."
    )
    
    print(f"\n--- Step 2: Detailed Financial Analysis ---")
    
    try:
        response_step2 = financial_analyst_executor.invoke({
            "input": task2_input,
            "chat_history": []
        })
        
        print(f"\n## Final Analysis Report ##")
        print("="*50)
        print(response_step2['output'])
        
    except Exception as e:
        print(f"Financial analysis failed: {str(e)}")
        print("Attempting direct analysis with sample stocks...")
        
        # Fallback: Direct analysis of a few known penny stocks
        sample_tickers = ["ZEEL.NS", "SUZLON.NS", "YESBANK.NS"]
        print(f"\nDirect analysis of sample tickers: {sample_tickers}")
        
        for ticker in sample_tickers:
            try:
                data = ImprovedYFinanceTool.get_stock_info(ticker)
                print(f"\n{ticker} Analysis:")
                print(f"Company: {data.get('company_name', 'N/A')}")
                print(f"Current Price: ₹{data.get('current_price_inr', 'N/A')}")
                print(f"52W High: ₹{data.get('52_week_high_inr', 'N/A')} ({data.get('high_date', 'N/A')})")
                print(f"52W Low: ₹{data.get('52_week_low_inr', 'N/A')} ({data.get('low_date', 'N/A')})")
                print(f"Market Cap: ₹{data.get('market_cap_crore', 'N/A')} Cr")
                print(f"Recent Trend: {data.get('recent_trend', 'N/A')}")
                if 'error' in data:
                    print(f"Error: {data['error']}")
                print("-" * 30)
                
            except Exception as ticker_error:
                print(f"Failed to analyze {ticker}: {str(ticker_error)}")

if __name__ == '__main__':
    run_stock_analysis()