import sys
import asyncio
import logging
from typing import List, Dict, Any

from src.agent_graph import create_agent_graph
from src.schemas.state import GraphState
from src.config import logger

# Configure logging to file for CLI to avoid cluttering output
file_handler = logging.FileHandler("medagent.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
# logger.setLevel(logging.INFO) # Already set in config

async def run_cli():
    """
    Main entry point for the Command Line Interface.
    """
    graph = create_agent_graph()

    # In-memory chat history for the CLI session
    chat_history: List[Dict[str, str]] = []

    print("🏥 Medical Research Assistant (CLI)")
    print("Enter your medical query (or 'exit' to quit).")
    print("Type 'history' to view conversation history.\n")

    while True:
        try:
            user_query = input("\n> ").strip()
            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Show conversation history
            if user_query.lower() == "history":
                if not chat_history:
                    print("\n📭 No conversation history yet.")
                else:
                    print(f"\n📜 Conversation History ({len(chat_history)} turns):")
                    print("─" * 60)
                    for i, turn in enumerate(chat_history, 1):
                        print(f"\n  [{i}] 🧑 Query: {turn['query']}")
                        # Show first 200 chars of response
                        response_preview = turn["response"][:200]
                        if len(turn["response"]) > 200:
                            response_preview += "..."
                        print(f"      🤖 Response: {response_preview}")
                    print("\n" + "─" * 60)
                continue

            print("\nProcessing...")
            logger.info(f"CLI User Query: {user_query}")

            # Build conversation history context for the agents
            history_context = [
                {"query": h["query"], "response": h["response"]}
                for h in chat_history
            ]

            initial_state = {
                "userQuery": user_query,
                "messages": [],
                "conversationHistory": history_context,
                "tasks": {},
                "medILlamaResponse": "",
                "webSearchResponse": "",
                "finalResponse": "",
                "iterationCount": 0,
                "qualityPassed": True,
                "requiredAgents": {
                    "medILlama": False,
                    "webSearch": False,
                    "rag": False,
                },
                "isSimpleQuery": False,
            }

            final_response = ""

            async for event in graph.astream(
                initial_state, stream_mode="updates"
            ):
                for node, update in event.items():
                    # print(f"\n--- Update from {node} ---") # Optional: reduce verbosity for user
                    if "finalResponse" in update:
                        final_response = update["finalResponse"]
                        print(f"\n📝 Final Response:\n{final_response}")
                    elif "orchestrationData" in update:
                         # Show plan
                         orch_data = update.get("orchestrationData")
                         plan = getattr(orch_data, "plan", "No plan") if hasattr(orch_data, "plan") else orch_data.get("plan")
                         print(f"📋 Plan: {plan}")

            # Save to chat history
            if final_response:
                chat_history.append(
                    {"query": user_query, "response": final_response}
                )
                logger.info("Response generated and saved.")
            else:
                 logger.warning("No final response generated.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"CLI Error: {e}", exc_info=True)

    # Show summary on exit
    if chat_history:
        print(f"\n📊 Session summary: {len(chat_history)} queries processed.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
    asyncio.run(run_cli())
