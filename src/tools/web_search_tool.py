from langchain_community.tools import TavilySearchResults

def build_web_search_tool():
    return TavilySearchResults(
        max_results=5,
        search_depth="basic"
    )
