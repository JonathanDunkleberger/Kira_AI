# web_search.py - Handles web search functionality.

import asyncio
from googleapiclient.discovery import build # Requires google-api-python-client
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID

def GoogleSearch(query: str, num_results: int = 3) -> str:
    """Performs a Google search and returns formatted results."""
    print(f"   Performing web search for: {query}")
    try:
        # Check if API keys are set up before building service
        if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("YOUR_"):
            print("   ERROR: Google API Key not set in config.py. Web search disabled.")
            return "Web search is not configured."
        if not GOOGLE_CSE_ID or GOOGLE_CSE_ID.startswith("YOUR_"):
            print("   ERROR: Google CSE ID not set in config.py. Web search disabled.")
            return "Web search is not configured."

        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
        
        if 'items' not in res or not res['items']:
            return "No search results found."

        snippets = []
        for i, item in enumerate(res['items']):
            # Limit snippet length for LLM context, avoid very long results
            snippet_text = item.get('snippet', '')
            if len(snippet_text) > 200: # Trim long snippets
                snippet_text = snippet_text[:200] + "..."
            snippets.append(f"[{i+1}] {item.get('title', 'No Title')}: {snippet_text}")
        
        return "\n".join(snippets)

    except Exception as e:
        print(f"   Error during web search: {e}")
        return "There was an error searching the web."

async def async_GoogleSearch(query: str, num_results: int = 3) -> str:
    """Asynchronous wrapper for the Google Search function."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, GoogleSearch, query, num_results)