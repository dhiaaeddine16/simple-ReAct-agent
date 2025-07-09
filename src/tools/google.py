from src.config.logging import logger
from typing import List, Dict
from googlesearch import search as google_search
from langchain_community.document_loaders import WebBaseLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re


def fetch_title_snippet(url: str) -> Dict[str, str]:
    """
    Load the page using WebBaseLoader and extract the title and snippet.

    Parameters:
    -----------
    url : str
        The URL of the web page.

    Returns:
    --------
    Dict[str, str]
        A dictionary with 'title' and 'snippet'.
    """
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        bad_chars = ['"', "\n", "'"]
        text = documents[0].page_content
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        for j in bad_chars:
            if j == '\n':
                text = text.replace(j, ' ')
            else:
                text = text.replace(j, '')
        title = documents[0].metadata.get("title", "")
        if documents:
            return {
                "title": title,
                "snippet": text[:500]  # first 300 chars as snippet
            }
        else:
            return {"title": "", "snippet": ""}
    except Exception as e:
        logger.warning(f"Failed to load or parse {url}: {e}")
        return {"title": "", "snippet": ""}

def format_top_search_results(urls: List[str], max_workers: int = 5) -> List[Dict[str, str]]:
    """
    Format the list of URLs into a list of dictionaries with position, link, title, and snippet using multithreading.
    """
    results = []

    def process(index_url):
        i, url = index_url
        try:
            meta = fetch_title_snippet(url)
        except Exception as e:
            logger.warning(f"Thread error on {url}: {e}")
            meta = {"title": "", "snippet": ""}
        return {
            "position": i + 1,
            "link": url,
            "title": meta["title"],
            "snippet": meta["snippet"]
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process, (i, url)) for i, url in enumerate(urls)]
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results to preserve original order
    results.sort(key=lambda x: x["position"])
    return results

def search(search_query: str, num_results: int = 5) -> str:
    """
    Perform a Google search using the googlesearch-python library and return the top results as a JSON string.

    Parameters:
    -----------
    search_query : str
        The search query to execute.
    num_results : int, optional
        Number of top search results to retrieve (default is 10).

    Returns:
    --------
    str
        A JSON string containing the top search results or an error message.
    """
    try:
        # Retrieve URLs from Google
        urls = list(google_search(search_query, num_results=num_results))

        # Format and return as JSON
        formatted = format_top_search_results(urls)
        return json.dumps({"top_results": formatted}, indent=2)

    except Exception as e:
        # Log and return error
        logger.error(f"Google search failed: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    query = "Best football players in Barcelona, Spain"
    print(search(query))
