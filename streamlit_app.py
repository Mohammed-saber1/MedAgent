"""
MedAgent — Streamlit Chat Interface with Streaming Output
Run:  streamlit run streamlit_app.py
"""

import asyncio
import uuid
import time
import threading
import streamlit as st

from src.agent_graph import create_agent_graph
from src.session_manager import session_manager, SessionManager


from streamlit.runtime.scriptrunner import add_script_run_ctx

def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code.
    Uses a dedicated thread with its own event loop to avoid
    conflicts with uvloop / Streamlit's internal loop.
    """
    result = [None]
    exception = [None]

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result[0] = loop.run_until_complete(coro)
        except Exception as exc:
            exception[0] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_target)
    add_script_run_ctx(thread)
    thread.start()
    thread.join()

    if exception[0] is not None:
        raise exception[0]
    return result[0]


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="MedAgent — Medical Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: #0E1117;
    border-right: 1px solid #1a1f2e;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: white; }

.stChatMessage [data-testid="stMarkdownContainer"] {
    font-size: 0.95rem;
    line-height: 1.65;
}

div[data-testid="stStatusWidget"] {
    border-radius: 12px;
    border: 1px solid #2a2f3e;
}

.header-strip { padding: 1rem 0 0.5rem; text-align: center; }
.header-strip h1 {
    background: linear-gradient(135deg, #6C63FF, #48C9B0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700; font-size: 1.8rem; margin: 0;
}
.header-strip p { color: #8899aa; margin: 0.25rem 0 0.5rem; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state defaults ───────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = create_agent_graph()
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False


# ── Helpers ──────────────────────────────────────────────
def new_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.history_loaded = False


def load_history():
    """Load existing history from MongoDB for the current session."""
    async def _load():
        sm = SessionManager()
        try:
            session = await sm.get_session(st.session_state.session_id)
            if session and session.get("history"):
                st.session_state.messages = []
                for entry in session["history"]:
                    st.session_state.messages.append(
                        {"role": "user", "content": entry["query"]}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": entry["response"]}
                    )
        finally:
            sm.close()

    run_async(_load())
    st.session_state.history_loaded = True


# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAgent")
    st.caption("Multi-Agent Medical Research Assistant")
    st.divider()

    if st.button("➕  New Session", use_container_width=True):
        new_session()
        st.rerun()

    with st.expander("Session ID"):
        st.code(st.session_state.session_id, language=None)

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Your query is evaluated (simple / complex)\n"
        "2. Complex queries are decomposed into sub-tasks\n"
        "3. Specialized agents research in parallel\n"
        "4. Results are compiled into a streamed response"
    )
    st.divider()
    st.caption("⚠️ For research & educational purposes only.")


# ── Header ───────────────────────────────────────────────
st.markdown(
    '<div class="header-strip">'
    "<h1>🏥 Medical Research Assistant</h1>"
    "<p>Ask any medical question — powered by multi-agent AI with real-time streaming</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Load history from MongoDB on first render ────────────
if not st.session_state.history_loaded:
    load_history()

# ── Display existing messages ────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"],
        avatar="🧑‍⚕️" if msg["role"] == "user" else "🤖",
    ):
        st.markdown(msg["content"])


# ── Process user input ───────────────────────────────────
def run_agent(user_query: str) -> str:
    """Run the agent graph, show progress, stream the response."""
    graph = st.session_state.graph

    # Container to hold results from the async thread
    results = {"final_response": "", "is_simple": False}

    
    # Create UI elements
    # We need to create the chat message container first to stream into it
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status("🔍 Analyzing...", expanded=True)
        answer_placeholder = st.empty()
    
    async def _run():
        sm = SessionManager()
        try:
            # Ensure session
            session = await sm.get_session(st.session_state.session_id)
            if session is None:
                await sm.create_session(st.session_state.session_id)
                session = await sm.get_session(st.session_state.session_id)

            history = [
                {"query": e["query"], "response": e["response"]}
                for e in (session.get("history") or [])
            ]

            initial_state = {
                "userQuery": user_query,
                "messages": [],
                "conversationHistory": history,
                "tasks": {},
                "medILlamaResponse": "",
                "webSearchResponse": "",
                "finalResponse": "",
                "iterationCount": 0,
                "qualityPassed": True,
                "requiredAgents": {
                    "medILlama": False, "webSearch": False, "rag": False
                },
                "isSimpleQuery": False,
            }
            # ── Run graph with status updates & streaming ────────────────
            final_response_accumulator = ""
            
            # Buffer for evaluate node to strip "SIMPLE: " prefix
            evaluate_buffer = ""
            is_streaming_simple = False
            
            # We use astream_events to catch both node updates (for status) and LLM tokens (for streaming)
            async for event in graph.astream_events(initial_state, version="v1"):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    # Check if this streaming event comes from the 'compile' or 'reflect' node
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    
                    if node_name == "compile" or node_name == "reflect":
                        status_container.update(label="📝 Generating response...", state="running")
                        content = event["data"]["chunk"].content
                        if content:
                            final_response_accumulator += content
                            answer_placeholder.markdown(final_response_accumulator + "▌")
                    
                    elif node_name == "evaluate":
                        # For evaluate, we only stream if it starts with SIMPLE:
                        content = event["data"]["chunk"].content
                        if content:
                            evaluate_buffer += content
                            
                            # Check if we should start streaming
                            if not is_streaming_simple:
                                if evaluate_buffer.startswith("SIMPLE:"):
                                    is_streaming_simple = True
                                    # Stream what we have so far, minus the prefix
                                    to_stream = evaluate_buffer[7:].lstrip()
                                    if to_stream:
                                        final_response_accumulator += to_stream
                                        answer_placeholder.markdown(final_response_accumulator + "▌")
                                        status_container.update(label="✅ Generating simple response...", state="running")
                                elif len(evaluate_buffer) > 7 and not evaluate_buffer.startswith("SIMPLE:"):
                                    # It's COMPLEX or something else, don't stream
                                    pass
                            else:
                                # We are already streaming a simple response
                                final_response_accumulator += content
                                answer_placeholder.markdown(final_response_accumulator + "▌")

                
                elif kind == "on_chain_end":
                    # Check for node completions to update status
                    node_name = event.get("name")
                    if node_name == "evaluate":
                        output = event["data"].get("output")
                        if output and output.get("isSimpleQuery"):
                            results["is_simple"] = True
                            if not final_response_accumulator: # Fallback if streaming failed
                                final_response_accumulator = output.get("finalResponse", "")
                                answer_placeholder.markdown(final_response_accumulator)
                            status_container.update(label="✅ Simple query", state="complete")
                        else:
                            status_container.update(label="🎵 Orchestrating agents...")
                            st.write("⚖️ Query classified as **complex**")

                    
                    elif node_name == "orchestrate":
                        st.write("🎵 Sub-tasks assigned")
                    elif node_name == "med_illama":
                        st.write("🏥 MedILlama analysis complete")
                    elif node_name == "web_search":
                        st.write("🔎 Web search results collected")
                    elif node_name == "pubmed_rag":
                        st.write("📚 PubMed RAG retrieval complete")
                    elif node_name == "compile":
                        status_container.update(label="✅ Analysis complete", state="complete")

            results["final_response"] = final_response_accumulator
            # Final update to remove cursor
            answer_placeholder.markdown(final_response_accumulator)

            # Persist
            await sm.add_to_history(
                session_id=st.session_state.session_id,
                query=user_query,
                response=results["final_response"],
                is_simple=results["is_simple"],
            )
        finally:
            sm.close()


    run_async(_run())

    final_response = results["final_response"]
    is_simple = results["is_simple"]
    
    return final_response





# ── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    # Run agent and get response
    response = run_agent(prompt)

    # Save to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

