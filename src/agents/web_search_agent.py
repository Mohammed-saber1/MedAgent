import os
import json
from langchain_tavily import TavilySearch
from src.schemas.state import GraphState, WebSearchResult, WebSearchResultItem
from src.utils.prompts import search_summary_prompt
from src.config import LLM, logger, settings

async def web_search_agent(state: GraphState) -> dict:
    """
    Retrieves and summarizes information from the web using Tavily.
    """
    logger.info("🔎 Web Search Agent Started")
    
    tasks = state.get("tasks", {}).get("WebSearch", [])
    web_search_tasks = []
    
    # Normalize tasks
    if not tasks and state.get("userQuery"):
        web_search_tasks.append(state["userQuery"])
    else:
        for t in tasks:
            if hasattr(t, 'query'):
                 web_search_tasks.append(t.query)
            elif isinstance(t, dict):
                 web_search_tasks.append(t.get('query'))
            else:
                 web_search_tasks.append(str(t))

    tavily_tool = TavilySearch(
        max_results=settings.MAX_SEARCH_RESULTS,
        tavily_api_key=settings.TAVILY_API_KEY
    )
    
    all_results = []
    
    for query in web_search_tasks:
        logger.info(f"🔎 Searching for: \"{query}\"")
        try:
            raw_results = await tavily_tool.ainvoke(query)

            # Map to our schema
            search_items = []

            # Normalize: new TavilySearch may return a dict
            # with a "results" key, a list, or a string.
            result_list = []
            if isinstance(raw_results, dict):
                result_list = raw_results.get("results", [])
            elif isinstance(raw_results, list):
                result_list = raw_results
            elif isinstance(raw_results, str):
                # String response — wrap it as a single item
                result_list = [{"url": "", "title": "", "content": raw_results}]

            for r in result_list:
                if isinstance(r, dict):
                    search_items.append(WebSearchResultItem(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        content=r.get("content", ""),
                    ))
            
            logger.info(f"✓ Found {len(search_items)} results for query: \"{query}\"")
            
            all_results.append(WebSearchResult(
                query=query,
                results=search_items
            ))
            
        except Exception as e:
             logger.error(f"❌ Error searching for \"{query}\": {e}")

    if not all_results:
        logger.warning("❌ No search results found for any query")
        return {} # Return empty update, don't change state

    # Prepare content for summary
    combined_content = ""
    for r in all_results:
        combined_content += f"Query: {r.query}\n"
        for item in r.results:
            combined_content += f"Source: {item.url}\n{item.content}\n\n"
        combined_content += "---\n\n"

    # Token limit check (rough optimization)
    if len(combined_content) / 4 > 6000:
        logger.warning("⚠️ Content too large for summary, relying on LLM context window limits")

    try:
        logger.info("📝 Generating summary of search results...")
        chain = search_summary_prompt | LLM
        
        # Extract URLs for prompt
        urls_list = []
        for r in all_results:
            for item in r.results:
                urls_list.append({"query": r.query, "url": item.url})

        summary = await chain.ainvoke({
            "searchResults": combined_content,
            "urls": json.dumps(urls_list)
        })
        
        logger.info("✅ Web Search Agent Completed")
        
        return {
            "webSearchResponse": summary.content,
            "webSearchResults": all_results
        }

    except Exception as e:
        logger.error(f"❌ Error generating summary: {e}")
        return {
            "webSearchResponse": f"Error generating search summary: {str(e)}",
            "webSearchResults": all_results
        }
