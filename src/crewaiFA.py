import os
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool # Your existing import for custom tool

# Agent and Agent Executor
from langchain.agents import AgentExecutor, create_openai_tools_agent # Or other agent types like create_react_agent

# Prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Utilities (optional, for more complex state or message history management)
# from langchain.memory import ChatMessageHistory

# Load environment variables
load_dotenv()

# --- 1. LLM Configuration ---
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="gemma2-9b-it"
)

# --- 2. Tool Definitions ---
# Web Search Tool
web_search_tool = DuckDuckGoSearchRun()

# Custom YFinance Tool (Your existing definition)
import yfinance as yf # Make sure yf is imported

class YFinanceTool():
    @staticmethod
    def get_stock_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            recommendations = stock.recommendations
            current_price = info.get("currentPrice", info.get("previousClose"))
            fifty_two_week_high = info.get("fiftyTwoWeekHigh")
            fifty_two_week_low = info.get("fiftyTwoWeekLow")
            
            return {
                "ticker": ticker,
                "current_price_inr": current_price,
                "52_week_high_inr": fifty_two_week_high,
                "52_week_low_inr": fifty_two_week_low,
                "company_name": info.get("longName"),
                "recommendations": recommendations.tail().to_dict('records') if recommendations is not None and not recommendations.empty else "N/A",
                "recent_news": [news['title'] for news in stock.news[:3]] if stock.news else "N/A"
            }
        except Exception as e:
            return f"Error fetching data for {ticker}: {str(e)}"

@tool("YahooFinanceStockDataTool") # Renamed slightly for clarity if needed, or keep your name
def yfinance_stock_data_tool(ticker: str) -> dict:
    """
    Fetches stock information for a given ticker using Yahoo Finance.
    Input should be a stock ticker symbol (e.g., 'RELIANCE.NS' for NSE or 'RELIANCE.BO' for BSE).
    Provides current price, 52-week high/low, company name, recent analyst recommendations, and recent news.
    All data relevant up to May 2025.
    """
    return YFinanceTool.get_stock_info(ticker)

# --- 3. Agent Definitions ---

# Agent 1: Web Search Agent
web_search_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert Web Researcher for Indian Financial Markets. Your current date context is May 23, 2025. "
        "Your goal is to find current and relevant information for financial analysis, focusing on the Indian stock market (BSE & NSE). "
        "Prioritize credible Indian financial portals and recent data (last 1-3 months before May 2025). "
        "Always cite sources and publication dates if possible when providing summaries."
        "You have access to a DuckDuckGo search tool."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # For intermediate steps if using ReAct or similar
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
    # handle_parsing_errors=True # Useful for debugging
)

# Agent 2: Financial Analyst Agent
financial_analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a specialist Indian Stock Market Data Analyst. Your current date context is May 23, 2025. "
        "Your goal is to analyze Indian penny stocks (BSE & NSE) based on provided information or tickers. "
        "You need to identify those that hit a notable high, then a significant low in the last 1-2 years (before May 2025), and are now showing strong recovery signs. "
        "You meticulously fetch and present stock prices (in INR), 52-week high/lows, historical trends, trading volumes, analyst ratings, and company-specific news using the YahooFinanceStockDataTool. "
        "Focus on data points indicating recovery and short-term growth potential up to May 2025. Present findings clearly, using tables for comparisons."
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
    # handle_parsing_errors=True
)

# --- 4. Orchestration (Sequential Example) ---
user_query_overall = (
    "Identify up to 5 Indian penny stocks listed on BSE or NSE which "
    "got to a low value after achieving a notable high in the last 1-2 years (before May 2025), "
    "and are now again showing strong potential for short-term growth. "
    "Focus on data and news up to May 2025."
)

# This is a simplified orchestration. A more robust solution might involve LangGraph.
if __name__ == '__main__':
    print("## Starting LangChain Agent Execution ##")
    
    # Step 1: Define 'penny stock' and get an initial list of tickers using the Web Search Agent
    # (This requires careful prompting for the web search agent)
    task1_input = (
        "First, define 'Indian penny stock' using common criteria for May 2025 "
        "(e.g., price < INR 50, or market cap < INR 500 crore). State the definition you will use. "
        "Then, using this definition, search the web for a list of 10-15 potential Indian penny stock ticker symbols "
        "(format: STOCK.NS or STOCK.BO) that might be interesting for further analysis based on the user's overall goal: "
        f"'{user_query_overall}'. Focus on finding tickers that might have experienced volatility."
    )
    print(f"\n--- Task 1: Defining Penny Stocks and Initial Scan (Web Search Agent) ---")
    print(f"Input: {task1_input}")
    
    # Initialize chat history for each agent if you want to maintain conversation context across multiple turns with the SAME agent
    # For a simple sequential pass, it might not be strictly necessary if each input is self-contained.
    web_search_history = [] 
    
    response_step1 = web_search_executor.invoke({
        "input": task1_input,
        "chat_history": web_search_history 
    })
    print(f"Web Search Agent Output:\n{response_step1['output']}")
    web_search_history.extend([
        HumanMessage(content=task1_input),
        AIMessage(content=response_step1['output'])
    ])

    # Step 2: Analyze the identified stocks using the Financial Analyst Agent
    # The output from step 1 needs to be parsed to extract the definition and tickers for step 2.
    # This parsing logic can be complex and might involve another LLM call or regex.
    # For this example, let's assume response_step1['output'] contains the list and definition.
    
    task2_input = (
        f"Based on the following information from web research: '{response_step1['output']}'\n\n"
        "Your task is to perform a detailed financial analysis. For each promising ticker mentioned, "
        "use the YahooFinanceStockDataTool to: \n"
        "1. Verify if it fits the pattern: notable high (last 1-2 yrs before May 2025), then significant drop, then recent recovery signs. \n"
        "2. Gather CMP (INR), 52-week high/low (INR), notable high & date, recent low & date. \n"
        "3. Look for evidence of recovery (price trends, volume, news via the tool). \n"
        "4. Note any analyst ratings/targets. \n"
        "Select up to 5 stocks that best fit the 'fallen angel with recovery potential' criteria. "
        "For each selected stock, provide: \n"
        "   a. Company Name and Ticker Symbol (with Exchange: BSE or NSE). \n"
        "   b. Current Market Price (CMP) in INR (as of May 2025). \n"
        "   c. The 'Notable High' price (INR) and approximate date/period. \n"
        "   d. The 'Recent Low' price (INR) and approximate date/period. \n"
        "   e. A brief explanation of the likely reasons for the price drop (if ascertainable from provided context or generic knowledge). \n"
        "   f. Clear evidence and explanation of current 'signs of recovery and potential'. \n"
        "   g. A concise investment thesis for short-term growth. \n"
        "   h. Any available analyst ratings or price targets in INR. If not available, state so. \n"
        "Present this as a structured report. Use tables for summaries where appropriate."
    )
    print(f"\n--- Task 2: Detailed Financial Analysis (Financial Analyst Agent) ---")
    # print(f"Input (will be long): {task2_input}") # Can be very verbose

    financial_analyst_history = []
    response_step2 = financial_analyst_executor.invoke({
        "input": task2_input,
        "chat_history": financial_analyst_history
    })
    
    print(f"\n## Final Report from Financial Analyst Agent: ##\n{response_step2['output']}")