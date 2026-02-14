import sys
import asyncio
from src.agent_graph import create_agent_graph


async def main():
    graph = create_agent_graph()

    # In-memory chat history for the CLI session
    chat_history = []

    print("🏥 Medical Research Assistant (Python)")
    print("Enter your medical query (or 'exit' to quit):")
    print("Type 'history' to view conversation history.\n")

    while True:
        try:
            user_query = input("\n> ")
            if user_query.lower() in ["exit", "quit"]:
                break

            if not user_query.strip():
                continue

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
                    print(f"\n--- Update from {node} ---")
                    if "finalResponse" in update:
                        final_response = update["finalResponse"]
                        print(f"\n📝 Final Response:\n{final_response}")
                    elif (
                        "medILlamaResponse" in update
                        and update["medILlamaResponse"]
                    ):
                        print("(MedILlama output received)")
                    elif (
                        "webSearchResponse" in update
                        and update["webSearchResponse"]
                    ):
                        print("(Web Search output received)")
                    elif "ragResponse" in update:
                        print("(RAG output received)")
                    elif "orchestrationData" in update:
                        orch_data = update["orchestrationData"]
                        plan = (
                            orch_data.get("plan")
                            if isinstance(orch_data, dict)
                            else getattr(orch_data, "plan", "No plan")
                        )
                        print(f"Plan: {plan}")

            # Save to chat history
            if final_response:
                chat_history.append(
                    {"query": user_query, "response": final_response}
                )

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    # Show summary on exit
    if chat_history:
        print(f"\n📊 Session summary: {len(chat_history)} queries processed.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
    asyncio.run(main())
