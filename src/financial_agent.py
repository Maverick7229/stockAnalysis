from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
# import openai
from dotenv import load_dotenv

load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")


# Web search agent to find information on the web
web_search_agent=Agent(
    name="web_search_agent",
    role="Search the web for information",
    model=Groq(id="mixtral-8x7b-32768"),
    tools=[
        DuckDuckGo(),
    ],
    instructions=[
        "Your goal is to find current and relevant information for financial analysis, especially for the Indian stock market.",
        "Always include the source URL and publication date of the information you find. Prioritize information from the last 3-6 months.",
        "When researching companies or stocks, look for recent news (e.g., earnings reports, new projects, regulatory changes, sector trends), analyst opinions, and market sentiment.",
        "Verify the credibility of sources. Prefer established financial news portals (e.g., Economic Times, Moneycontrol, Business Standard), official company announcements, and exchange websites (BSE, NSE).",
        "If asked for definitions (e.g., 'penny stock in India'), provide a comprehensive answer citing sources.",
        "Focus on information relevant to the query and published on or before today's date (May 23, 2025), unless historical data is specifically requested."
    ],
    # instructions=["Always include the source of the information you find."],
    show_tools_calls=True,
    markdown=True
)
 
# Financial agent to answer questions about finance and economics
financial_agent=Agent(
    name="financial_agent",
    role="Answer questions about finance and economics.",
    model=Groq(id="mixtral-8x7b-32768"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    # instructions=[" Use tables to present data when possible. Always state the past, current and predicted stockprices."],
    instructions=[
        "You are a specialist in Indian stock market data (BSE & NSE).",
        "Analyse and provide profit expense ratio report for the last 3 years.",
        "Analyse and provide revenue report for the last 3 years.",
        "Analyse and provide yearly report for the last 3 years.",
        "Provide the latest available stock data. Specify if data is not real-time and mention its timestamp if available (e.g., 'as of closing May 22, 2025').",
        "When providing stock prices (past, current), ensure they are in INR. Explicitly state 'INR'.",
        "For 'predicted prices', provide analyst target prices if available from YFinanceTools. If not, state that direct predictions are unavailable through this tool and provide qualitative outlooks if found.",
        "Focus on data points relevant to identifying stocks that have fallen from a high and are now recovering: 52-week high/low, historical price trends (e.g., 1-month, 3-month, 6-month changes), recent price movements, trading volumes, and company-specific news from YFinanceTools.",
        "Use tables to present comparative data clearly.",
        "Always specify the stock exchange (BSE or NSE) for each stock mentioned.",
        "Prioritize data from the last trading day or week. Confirm the date of the data provided."
    ],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_search_agent, financial_agent],
    description="Coordinates web research and financial data analysis to identify promising Indian stocks based on specific recovery criteria.", # Added description
    # model=Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key), # Added model for the orchestrator agent
    model=Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY")),
    instructions=[
        "You are an expert financial analyst specializing in the Indian stock market (BSE & NSE). Your current date context is May 23, 2025.",
        "Your main task is to identify up to 5 Indian penny stocks with significant short-term growth potential based on a specific recovery pattern.",
        "Follow these steps carefully:",
        "1. Define 'Indian penny stock'. You can use a common definition (e.g., stocks trading below INR 20 or INR 50, or with a market cap below INR 500 crore). State the definition you are using for this analysis. Use the `web_search_agent` if needed to find common definitions in the Indian context or to decide on a reasonable threshold.",
        "2. Search for penny stocks listed on the Bombay Stock Exchange (BSE) or National Stock Exchange (NSE) of India that fit your definition.",
        "3. For each identified penny stock, analyze if it previously reached a 'notable high' (e.g., a 52-week high or a significant peak in the last 1-2 years) and then experienced a 'significant price drop' (e.g., a decline of 40-60%+ from that high). Focus on lows reached in the past year, avoiding distant historical lows like the COVID crash unless it's the most recent significant dip from a peak.",
        "4. Filter these stocks to find those 'now exhibiting strong signs of recovery and potential for short-term growth (next 1-3 months)'. Use the `financial_agent` for stock data (current price, volume, fundamentals, recent news from YFinance) and the `web_search_agent` for recent news (last 1-3 months), sentiment, and catalyst events.",
        "5. 'Signs of recovery/potential' could include: sustained price increase in the recent 1-2 months, increased trading volume, positive company-specific news/announcements, sector-specific tailwinds, or favorable analyst commentary.",
        "6. For each of the final selected stocks (up to 5), provide:",
        "    a. Company Name and Ticker Symbol (with Exchange: BSE or NSE).",
        "    b. Current Market Price (CMP) in INR (ensure this is the latest available as of May 2025).",
        "    c. The 'Notable High' price (INR) and approximate date/period (within the last 1-2 years).",
        "    d. The 'Recent Low' price (INR) after the high and approximate date/period (within the last year).",
        "    e. A brief explanation of the likely reasons for the price drop (if ascertainable via web search).",
        "    f. Clear evidence and explanation of current 'signs of recovery and potential' (citing news or data).",
        "    g. A concise investment thesis for short-term growth.",
        "    h. Any available analyst ratings or price targets in INR from the `financial_agent`. If not available, state so.",
        "7. Present the information clearly. Use tables for the final list of stocks and their key metrics.",
        "8. Ensure all financial data is as current as possible (latest trading day before or on May 23, 2025). Always specify that prices are in INR.",
        "9. If you need to search the web, instruct the `web_search_agent` to prioritize recent (last 1-3 months) and reputable Indian financial sources.",
        "10. If you need financial data, instruct the `financial_agent` to fetch the latest available information for BSE/NSE listed stocks.",
        "11. Critically evaluate the information. Do not select stocks if convincing evidence for recovery and potential is lacking."
    ],
    show_tools_calls=True,
    markdown=True,
)

# multi_ai_agent.print_response("Analyze top 5 Indian stocks with strong growth potential in the next month. Compare performance metrics for leading Indian stocks. Show recent analyst ratings for Indian large-cap stocks", stream=True)
# multi_ai_agent.print_response("what are the top 5 Indian penny stocks with strong growth potential in the next month?", stream=True)
# multi_ai_agent.print_response("what are the top 5 Indian penny stocks which got to a low value after achieving high and are now again showing potential?", stream=True)
user_prompt = "what are the top 5 Indian penny stocks listed on BSE or NSE which got to a low value after achieving a notable high in the last 1-2 years, and are now again showing strong potential for short-term growth? Focus on data and news up to May 2025."
multi_ai_agent.print_response(user_prompt, stream=True)


