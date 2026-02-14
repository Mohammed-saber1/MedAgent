import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from src.schemas.state import GraphState, WebSearchResult, WebSearchResultItem
from src.utils.prompts import search_summary_prompt
from src.config import LLM

async def web_search_agent(state: GraphState) -> dict:
    """
    Retrieves and summarizes information from the web.
    """
    print("\n🔎 Web Search Agent Started")
    
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

    tavily_tool = TavilySearchResults(
        max_results=5,
        api_key=os.getenv("TAVILY_API_KEY")
    )
    
    all_results = []
    
    for query in web_search_tasks:
        print(f"\n🔎 Searching for: \"{query}\"")
        try:
            # tavily_tool.ainvoke returns a list of dictionaries usually
            raw_results = await tavily_tool.ainvoke(query)
            
            # Map to our schema
            search_items = []
            if isinstance(raw_results, list):
                for r in raw_results:
                    search_items.append(WebSearchResultItem(
                        url=r.get("url"),
                        title=r.get("title", ""), # title might not always be there depending on Tavily output format updates
                        content=r.get("content")
                    ))
            
            print(f"✓ Found {len(search_items)} results for query: \"{query}\"")
            
            all_results.append(WebSearchResult(
                query=query,
                results=search_items
            ))
            
        except Exception as e:
             print(f"❌ Error searching for \"{query}\": {e}")

    if not all_results:
        print("❌ No search results found for any query")
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
        print("⚠️ Content too large for summary, returning raw results only (or implementing truncation)")
        # For now, let's truncate context if needed or just proceed and rely on LLM context window (70b usually has large window)
        # combined_content = combined_content[:24000] 

    try:
        print("\n📝 Generating summary of search results...")
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
        
        print("\n✅ Web Search Agent Completed")
        
        return {
            "webSearchResponse": summary.content,
            "webSearchResults": all_results
        }

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return {
            "webSearchResponse": f"Error generating search summary: {str(e)}",
            "webSearchResults": all_results
        }
