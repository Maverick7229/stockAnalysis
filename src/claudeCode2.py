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

# --- Enhanced Company to Ticker Mapping ---
COMPANY_TICKER_MAP = {
    # From web search results
    'South Indian Bank': 'SOUTHBANK.NS',
    'Srichakra Cement': 'SRICHAKRA.NS', 
    'Omansh Enterprises': 'OMANSH.NS',
    'Veto Switchgear': 'VETO.NS',
    'Visaka Industries': 'VISAKA.NS',
    'Emami Paper Mills': 'EMAMIPAP.NS',
    'Easy Trip Planners': 'EASEMYTRIP.NS',
    'Infibeam Avenues': 'INFIBEAM.NS',
    'Filatex Fashions': 'FILATEX.NS',
    'Alstone Textiles': 'ALSTONE.NS',
    'Taparia Tools': 'TAPARIA.NS',
    
    # Known penny stocks
    'Zee Entertainment': 'ZEEL.NS',
    'Suzlon Energy': 'SUZLON.NS',
    'Yes Bank': 'YESBANK.NS',
    'Reliance Power': 'RPOWER.NS',
    'Reliance Communications': 'RCOM.NS',
    'Jet Airways': 'JETAIRWAYS.NS',
    'Jaiprakash Associates': 'JPASSOCIAT.NS',
    'PC Jeweller': 'PCJEWELLER.NS',
    'Vodafone Idea': 'IDEA.NS',
    'Swan Defence': 'RNAVAL.NS',
    'Bharti Infratel': 'BHARTIINFRA.NS',
    'GTL Infrastructure': 'GTLINFRA.NS',
    
    # Alternative names
    'Swan Defence and Heavy Industries': 'RNAVAL.NS',
    'PC Jeweller Limited': 'PCJEWELLER.NS',
    'Yes Bank Limited': 'YESBANK.NS',
    'Suzlon Energy Limited': 'SUZLON.NS',
}

# --- LLM Configuration ---
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="gemma2-9b-it"
)

# --- Enhanced Web Search Tool ---
web_search_tool = DuckDuckGoSearchRun()

@tool("structured_web_search")
def structured_web_search(query: str) -> str:
    """
    Enhanced web search that looks for Indian penny stocks and formats results.
    Returns structured information about companies and their potential ticker symbols.
    """
    try:
        results = web_search_tool.run(query)
        
        # Extract company names from results
        extracted_info = []
        
        # Look for companies in our mapping
        for company_name, ticker in COMPANY_TICKER_MAP.items():
            if company_name.lower() in results.lower():
                extracted_info.append(f"COMPANY: {company_name}\nTICKER: {ticker}\nSTATUS: Mentioned in search results\n")
        
        # Also look for any direct ticker mentions
        ticker_pattern = r'\b[A-Z][A-Z0-9]*\.(?:NS|BO)\b'
        found_tickers = re.findall(ticker_pattern, results)
        
        for ticker in found_tickers:
            extracted_info.append(f"TICKER: {ticker}\nSTATUS: Direct ticker found\n")
        
        formatted_result = results + "\n\n=== EXTRACTED STRUCTURED INFO ===\n" + "\n".join(extracted_info)
        return formatted_result
        
    except Exception as e:
        return f"Search failed: {str(e)}"

# --- Enhanced Extraction Functions ---
def extract_structured_tickers(search_text):
    """Extract tickers from structured search results"""
    tickers = []
    
    # Method 1: Look for TICKER: pattern
    ticker_lines = re.findall(r'TICKER:\s*([A-Z][A-Z0-9]*\.(?:NS|BO))', search_text)
    tickers.extend(ticker_lines)
    
    # Method 2: Direct pattern matching
    direct_tickers = re.findall(r'\b[A-Z][A-Z0-9]*\.(?:NS|BO)\b', search_text)
    tickers.extend(direct_tickers)
    
    # Method 3: Company name mapping
    for company_name, ticker in COMPANY_TICKER_MAP.items():
        if company_name.lower() in search_text.lower():
            tickers.append(ticker)
    
    # Remove duplicates and return
    return list(set(tickers))

def validate_tickers(tickers):
    """Validate tickers by trying to fetch basic info"""
    valid_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info and ('longName' in info or 'shortName' in info):
                valid_tickers.append(ticker)
                print(f"✓ {ticker} - Valid")
            else:
                print(f"✗ {ticker} - No info available")
        except Exception as e:
            print(f"✗ {ticker} - Error: {str(e)}")
    
    return valid_tickers

# --- Enhanced Agent Definition ---
enhanced_web_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert Indian Stock Market Researcher. Your task is to find penny stocks with recovery potential. "
        "When searching, focus on finding specific company names that match this criteria: "
        "1) Indian penny stocks (price < ₹50 or market cap < ₹500 crore) "
        "2) Experienced volatility in 2023-2024 "
        "3) Showing signs of recovery in 2025 "
        
        "For each company you identify, provide information in this EXACT format: "
        "COMPANY: [Full Company Name] "
        "TICKER: [Symbol].NS or [Symbol].BO "
        "STATUS: [Brief description of current status] "
        
        "Use the structured web search tool to find relevant information."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

enhanced_web_agent = create_openai_tools_agent(
    llm=llm,
    tools=[structured_web_search],
    prompt=enhanced_web_agent_prompt
)

enhanced_web_executor = AgentExecutor(
    agent=enhanced_web_agent,
    tools=[structured_web_search],
    verbose=True,
)

# --- Main execution function ---
def run_enhanced_ticker_extraction():
    print("## Enhanced Ticker Extraction System ##")
    
    # Step 1: Multiple targeted searches
    search_queries = [
        "Indian penny stocks recovery 2025 South Indian Bank Visaka Industries Emami Paper",
        "penny stocks volatility 2023 2024 Suzlon Zee Entertainment Yes Bank",
        "Indian small cap stocks fallen angel recovery Reliance Power Infibeam"
    ]
    
    all_found_tickers = []
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n--- Search {i}: {query[:50]}... ---")
        
        try:
            result = enhanced_web_executor.invoke({
                "input": f"Search for: {query}. Provide structured output with company names and ticker symbols.",
                "chat_history": []
            })
            
            search_output = result['output']
            print(f"Search Results Preview: {search_output[:300]}...")
            
            # Extract tickers from this search
            found_tickers = extract_structured_tickers(search_output)
            all_found_tickers.extend(found_tickers)
            print(f"Tickers found in this search: {found_tickers}")
            
        except Exception as e:
            print(f"Search {i} failed: {str(e)}")
    
    # Step 2: Remove duplicates and validate
    unique_tickers = list(set(all_found_tickers))
    print(f"\n--- All Unique Tickers Found: {unique_tickers} ---")
    
    # Step 3: Validate tickers
    print(f"\n--- Validating Tickers ---")
    valid_tickers = validate_tickers(unique_tickers)
    
    # Step 4: Add fallback tickers if needed
    fallback_tickers = ['ZEEL.NS', 'SUZLON.NS', 'YESBANK.NS', 'RPOWER.NS', 'RCOM.NS']
    
    if len(valid_tickers) < 5:
        print(f"\n--- Adding Fallback Tickers ---")
        for ticker in fallback_tickers:
            if ticker not in valid_tickers:
                valid_tickers.append(ticker)
                if len(valid_tickers) >= 10:
                    break
    
    print(f"\n--- Final Ticker List for Analysis ---")
    final_tickers = valid_tickers[:10]  # Limit to 10
    
    for i, ticker in enumerate(final_tickers, 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name = info.get('longName', info.get('shortName', 'Unknown'))
            current_price = info.get('currentPrice', info.get('previousClose', 'N/A'))
            print(f"{i}. {ticker} - {company_name} - ₹{current_price}")
        except:
            print(f"{i}. {ticker} - Unable to fetch details")
    
    return final_tickers

if __name__ == '__main__':
    final_tickers = run_enhanced_ticker_extraction()
    print(f"\n=== SUMMARY ===")
    print(f"Successfully identified {len(final_tickers)} tickers for analysis:")
    print(", ".join(final_tickers))