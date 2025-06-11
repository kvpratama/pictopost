import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.tools.tavily_search import TavilySearchResults

class LLMConfig:
    DEFAULT = "gemini-2.0-flash"
    VERSATILE = "gemini-2.0-flash-lite"
    CREATIVE = "gemini-2.0-flash-lite"
    GEMMA12B = "gemma-3-12b-it"
    GEMMA27B = "gemma-3-27b-it"

def create_llm(model_name="gemini-2.0-flash", temperature=0, google_api_key=None):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=google_api_key)

def get_default_llm(google_api_key):
    return create_llm(LLMConfig.DEFAULT, temperature=0, google_api_key=google_api_key)

def get_versatile_llm(google_api_key):
    return create_llm(LLMConfig.VERSATILE, temperature=0.5, google_api_key=google_api_key)

def get_creative_llm(google_api_key):
    return create_llm(LLMConfig.CREATIVE, temperature=1.0, google_api_key=google_api_key)

def get_gemma12b_llm(google_api_key):
    return create_llm(LLMConfig.GEMMA12B, temperature=1, google_api_key=google_api_key)

def get_gemma27b_llm(google_api_key):
    return create_llm(LLMConfig.GEMMA27B, temperature=0.5, google_api_key=google_api_key)

# def get_tavily_search(tavily_api_key):
#     os.environ["TAVILY_API_KEY"] = tavily_api_key
#     return TavilySearchResults(max_results=3)
    