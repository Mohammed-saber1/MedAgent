# 🏥 Medical Research Assistant

A **multi-agent AI system** designed to answer complex medical queries with accuracy, depth, and up-to-date information. It dynamically orchestrates specialized AI agents — each with a unique role — to decompose, research, compile, and quality-check medical responses using real-time web data and domain-specific medical models.

> **⚠️ Disclaimer**  
> This system is intended for **research and educational purposes only**. Its outputs are *not* professional medical advice. Always consult a licensed healthcare provider for medical decisions.

---

## 📑 Table of Contents

- [Core Concepts](#-core-concepts)
- [Architecture Overview](#-architecture-overview)
- [Agent Workflow Flowchart](#-agent-workflow-flowchart)
- [Sequence Diagram](#-sequence-diagram)
- [Detailed Agent Descriptions](#-detailed-agent-descriptions)
- [Iteration & Feedback Loop](#-iteration--feedback-loop)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Example Queries](#-example-queries)
- [Roadmap](#-roadmap)
- [License & Disclaimer](#-license--disclaimer)

---

## 🧠 Core Concepts

### What Problem Does This Solve?

Medical queries are rarely simple. A question like *"Compare SGLT2 inhibitors vs GLP-1 receptor agonists for Type 2 Diabetes"* requires:
- **Domain expertise** (clinical knowledge of drug mechanisms)
- **Current research** (latest clinical trials, FDA approvals)
- **Literature review** (PubMed abstracts, peer-reviewed studies)
- **Quality assurance** (fact-checking against medical guidelines)

No single AI model excels at all of these. This system solves the problem by **decomposing complex queries into sub-tasks** and assigning each to a specialized agent, then synthesizing the results into a comprehensive, cited report.

### Key Design Principles

| Principle | Description |
|---|---|
| **Multi-Agent Architecture** | Each agent has a single responsibility (evaluation, search, analysis, compilation, quality control) |
| **Parallel Execution** | Agents run concurrently via LangGraph, minimizing total response time |
| **Iterative Refinement** | A reflection loop detects gaps and triggers re-research cycles (up to 3 iterations) |
| **Dual-LLM Strategy** | Groq (Llama 3.3 70B) for orchestration & compilation; Ollama (MedLlama) for domain-specific medical analysis |
| **Evidence-Based** | Web search and PubMed RAG provide cited, verifiable references |

---

## 🏗 Architecture Overview

The system is built around a **LangGraph state machine** that routes a user query through a pipeline of specialized agents. The workflow is defined as a directed graph with conditional edges that enable branching (simple vs. complex queries) and looping (reflection feedback).

```
User Query
    │
    ▼
┌──────────────┐    Simple    ┌──────────────┐
│  Evaluation  │─────────────▶│  Direct      │──▶ Final Answer
│  Agent       │              │  Response    │
└──────┬───────┘              └──────────────┘
       │ Complex
       ▼
┌──────────────┐
│ Orchestration│───── Decomposes query into sub-tasks
│ Agent        │───── Assigns agents & generates plan
└──────┬───────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  MedILlama   │  │  Web Search  │  │  PubMed RAG  │
│  Agent       │  │  Agent       │  │  Agent       │
│ (Ollama)     │  │ (Tavily)     │  │ (Biopython)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │   Compile    │──▶ Synthesizes all outputs
                  │   Agent      │    into a unified report
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐    Pass
                  │  Reflection  │──────────▶ Final Answer
                  │  Agent       │
                  └──────┬───────┘
                         │ Fail (feedback)
                         │
                         └──────▶ Back to Orchestration
                                  (up to 3 iterations)
```

---

## 📊 Agent Workflow Flowchart

```mermaid
flowchart TD
    A["🧑 User Query"] --> B["⚖️ Evaluation Agent"]
    B -->|"SIMPLE"| C["✅ Direct Answer"]
    B -->|"COMPLEX"| D["🎵 Orchestration Agent"]

    D --> E{"Assign Agents"}
    E -->|"medILlama: true"| F["🏥 MedILlama Agent"]
    E -->|"webSearch: true"| G["🔎 Web Search Agent"]
    E -->|"rag: true"| H["📚 PubMed RAG Agent"]

    F --> I["📝 Compile Agent"]
    G --> I
    H --> I

    I --> J["🤔 Reflection Agent"]
    J -->|"Quality Passed ✅"| K["✅ Final Response"]
    J -->|"Quality Failed ❌"| L{"Iteration < MAX?"}
    L -->|"Yes"| D
    L -->|"No"| K

    style A fill:#4A90D9,color:#fff
    style B fill:#F5A623,color:#fff
    style C fill:#7ED321,color:#fff
    style D fill:#9013FE,color:#fff
    style F fill:#D0021B,color:#fff
    style G fill:#4A90D9,color:#fff
    style H fill:#50E3C2,color:#000
    style I fill:#F8E71C,color:#000
    style J fill:#BD10E0,color:#fff
    style K fill:#7ED321,color:#fff
```

---

## 🔄 Sequence Diagram

```mermaid
sequenceDiagram
    participant U as 🧑 User
    participant EV as ⚖️ Evaluation Agent
    participant OR as 🎵 Orchestration Agent
    participant ML as 🏥 MedILlama Agent
    participant WS as 🔎 Web Search Agent
    participant RAG as 📚 PubMed RAG
    participant CO as 📝 Compile Agent
    participant RE as 🤔 Reflection Agent

    U->>EV: Submit medical query
    EV->>EV: Classify query (SIMPLE / COMPLEX)

    alt Simple Query
        EV-->>U: Return direct answer
    else Complex Query
        EV->>OR: Forward query for decomposition
        OR->>OR: Decompose into sub-tasks
        OR->>OR: Identify required agents

        par Parallel Agent Execution
            OR->>ML: MedILlama sub-tasks
            ML->>ML: Generate medical analysis via Ollama
            ML-->>CO: Return medILlamaResponse
        and
            OR->>WS: Web Search sub-tasks
            WS->>WS: Search via Tavily API
            WS->>WS: Summarize results via Groq LLM
            WS-->>CO: Return webSearchResponse
        and
            OR->>RAG: PubMed RAG sub-tasks
            RAG->>RAG: Search PubMed via Biopython
            RAG->>RAG: Embed & retrieve via ChromaDB
            RAG-->>CO: Return ragResponse
        end

        CO->>CO: Merge all agent outputs
        CO->>CO: Generate comprehensive report
        CO->>RE: Send compiled response

        RE->>RE: Quality check response
        alt Quality Passed
            RE-->>U: Return final response
        else Quality Failed (iteration < MAX)
            RE->>OR: Send reflection feedback
            Note over OR,CO: Improvement cycle begins
            OR->>OR: Revise tasks based on feedback
        end
    end
```

---

## 🤖 Detailed Agent Descriptions

### 1. Evaluation Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/evaluation_agent.py` |
| **LLM** | Groq (Llama 3.3 70B) |
| **Purpose** | Classify queries as SIMPLE or COMPLEX |

- **Simple queries** (e.g., *"What is hypertension?"*) receive an immediate, direct answer without invoking other agents.
- **Complex queries** (e.g., *"Compare treatments for T2D with cardiovascular risk"*) are forwarded to the Orchestration Agent for multi-agent processing.
- Uses `queryEvaluationPrompt` with few-shot examples to classify accurately.

### 2. Orchestration Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/orchestration_agent.py` |
| **LLM** | Groq (Llama 3.3 70B) with structured output |
| **Purpose** | Decompose queries and assign sub-tasks to agents |

- Analyzes the query and produces a `DecompositionOutput` containing:
  - `tasks.MedILlama` — Sub-queries for domain-specific analysis
  - `tasks.Web` — Sub-queries for web search
  - `requiredAgents` — Boolean flags for each agent
- On **reflection failure**, uses an `improvementPrompt` to revise tasks based on feedback.

### 3. MedILlama Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/medillama_agent.py` |
| **LLM** | Ollama (MedLlama2 / custom medical model) |
| **Purpose** | Domain-specific medical analysis |

- Processes sub-tasks using a fine-tuned medical model running locally via Ollama.
- Provides detailed explanations of pathophysiology, treatment mechanisms, drug interactions, and clinical considerations.
- Output is structured for integration with other agent responses.

### 4. Web Search Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/web_search_agent.py` |
| **LLM** | Groq (for summarization) |
| **Search API** | Tavily |
| **Purpose** | Retrieve and summarize current medical research |

- Executes web searches for each sub-task using Tavily API (up to 5 results per query).
- Summarizes findings using a structured prompt covering: Overview, Detailed Findings, Clinical Implications.
- Preserves source URLs for citation in the final report.

### 5. PubMed RAG Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/pubmed_rag_agent.py` |
| **Tools** | Biopython (Entrez), ChromaDB, Ollama Embeddings |
| **Purpose** | Retrieve and analyze PubMed abstracts |

- Searches PubMed via NCBI's Entrez API (Biopython) for relevant abstracts.
- Splits documents using `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap).
- Creates an in-memory ChromaDB vector store with Ollama embeddings.
- Retrieves top-k most relevant chunks for each query.

### 6. Compile Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/compile_agent.py` |
| **LLM** | Groq (Llama 3.3 70B) |
| **Purpose** | Synthesize all agent outputs into a final report |

- Merges MedILlama analysis, Web Search evidence, and RAG context into a unified, well-structured report.
- Uses MLA-style citations with numbered references.
- If reflection feedback exists, uses a refinement prompt to improve the previous response.
- Selects appropriate prompts based on available data (with/without web results).

### 7. Reflection Agent
| Property | Detail |
|---|---|
| **File** | `src/agents/reflection_agent.py` |
| **LLM** | Ollama (MedLlama2 / custom medical model) |
| **Purpose** | Quality assurance and feedback |

- Reviews the compiled response for:
  - Medical inaccuracies or outdated information
  - Critical missing details
  - Terminological errors
  - Inconsistencies or potentially harmful advice
  - Knowledge gaps
- Returns `qualityPassed: true/false` with optional improvement feedback.
- Can be bypassed via `BYPASS_REFLECTION` config flag.

---

## 🔁 Iteration & Feedback Loop

The system implements an **iterative refinement cycle** to ensure response quality:

```mermaid
flowchart LR
    A["Orchestration"] --> B["Agents gather data"]
    B --> C["Compile response"]
    C --> D["Reflection check"]
    D -->|"Pass"| E["✅ Final Answer"]
    D -->|"Fail + feedback"| F["Increment iteration"]
    F -->|"iteration ≤ 3"| A
    F -->|"iteration > 3"| E

    style E fill:#7ED321,color:#fff
```

| Iteration | What Happens |
|---|---|
| **1st pass** | Initial decomposition → parallel agent execution → compile → reflect |
| **2nd pass** (if needed) | Orchestration reads feedback → revises/adds tasks → agents fill gaps → recompile → re-reflect |
| **3rd pass** (max) | Final attempt to resolve quality issues. If still failing, returns best-effort response |

The `MAX_ITERATIONS` constant (default: `3`) prevents infinite loops.

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Workflow Engine** | LangGraph | Directed graph for agent orchestration with conditional edges |
| **Prompt Framework** | LangChain | Modular prompt templates and structured output parsing |
| **Primary LLM** | Groq (Llama 3.3 70B) | Orchestration, web search summarization, compilation |
| **Medical LLM** | Ollama (MedLlama2) | Domain-specific medical analysis and quality reflection |
| **Web Search** | Tavily API | Real-time medical research retrieval |
| **Literature Search** | Biopython (Entrez) | PubMed abstract retrieval via NCBI |
| **Vector Store** | ChromaDB | In-memory embedding storage for RAG |
| **Embeddings** | Ollama Embeddings | Text vectorization for similarity search |
| **API Framework** | FastAPI | REST and WebSocket endpoints |
| **Server** | Uvicorn | ASGI server with hot reload |
| **Data Validation** | Pydantic | Schema validation for state and structured outputs |
| **Language** | Python 3.11 | Core implementation language |

---

## 📁 Project Structure

```
Medical-Research-Assistant/
│
├── src/
│   ├── agents/                    # Agent implementations
│   │   ├── evaluation_agent.py    # Query classification (simple/complex)
│   │   ├── orchestration_agent.py # Task decomposition and agent assignment
│   │   ├── medillama_agent.py     # Domain-specific medical analysis
│   │   ├── web_search_agent.py    # Tavily web search + summarization
│   │   ├── pubmed_rag_agent.py    # PubMed RAG with ChromaDB
│   │   ├── compile_agent.py       # Response synthesis and formatting
│   │   └── reflection_agent.py    # Quality assurance and feedback
│   │
│   ├── schemas/                   # Pydantic data models
│   │   ├── state.py               # GraphState, OrchestrationData, RequiredAgents
│   │   └── decomposition.py       # DecompositionOutput, Task, TasksByType
│   │
│   ├── utils/
│   │   └── prompts.py             # All LangChain prompt templates
│   │
│   ├── server/
│   │   └── app.py                 # FastAPI server (REST + WebSocket + Sessions)
│   │
│   ├── agent_graph.py             # LangGraph workflow definition
│   ├── config.py                  # LLM initialization and constants
│   ├── session_manager.py         # In-memory session store for API
│   └── main.py                    # CLI entry point (with chat history)
│
├── examples/
│   └── run_query.py               # Programmatic usage examples
│
├── tests/                         # Test suite
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── run_server.py                  # Server startup script
└── README.md                      # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.11+**
- **Conda** (recommended) or `venv`
- **Ollama** installed and running locally ([Download Ollama](https://ollama.com/))
- **API Keys**:
  - [Groq API Key](https://console.groq.com/) — for the primary LLM
  - [Tavily API Key](https://tavily.com/) — for web search

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Medical-Research-Assistant.git
cd Medical-Research-Assistant
```

### Step 2: Create a Conda Environment

```bash
conda create -n medical-research python=3.11 -y
conda activate medical-research
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OLLAMA_MODEL=medllama2
OLLAMA_BASE_URL=http://localhost:11434
NCBI_EMAIL=your_email@example.com
```

### Step 5: Prepare the Medical Model

```bash
ollama pull medllama2
ollama serve
```

> **Note:** You can use any Ollama-compatible medical model. Update `OLLAMA_MODEL` in `.env` accordingly.

---

## 💻 Usage

### CLI Mode (Interactive)

```bash
python -m src.main
```

This launches an interactive terminal with **built-in chat history**:

```
🏥 Medical Research Assistant (Python)
Enter your medical query (or 'exit' to quit):
Type 'history' to view conversation history.

> What are the latest treatments for Alzheimer's disease?

Processing...
⚖️ Evaluation Agent Started
🎵 Orchestration Agent Started
🏥 MedILlama Agent Started
🔎 Web Search Agent Started
📝 Compile Agent Started
🤔 Reflection Agent Started

📝 Final Response:
[Comprehensive medical report with citations...]

> history

📜 Conversation History (1 turns):
────────────────────────────────────────────────────────────
  [1] 🧑 Query: What are the latest treatments for Alzheimer's disease?
      🤖 Response: Alzheimer's disease treatment has evolved...
────────────────────────────────────────────────────────────
```

**CLI Commands:**
| Command | Description |
|---|---|
| `history` | View all past queries and responses from the current session |
| `exit` / `quit` | Exit the application (shows session summary) |

### Server Mode (FastAPI)

```bash
python run_server.py
```

The server starts on `http://localhost:8080` with:
- **Stateless query**: `POST /api/query`
- **Session management**: `POST/GET/DELETE /api/sessions`
- **Session query**: `POST /api/sessions/{id}/query`
- **Health check**: `GET /health`

---

## ⚙️ Configuration

Configuration is managed via `src/config.py` and environment variables:

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | API key for Groq cloud LLM | *Required* |
| `TAVILY_API_KEY` | API key for Tavily web search | *Required* |
| `OLLAMA_MODEL` | Ollama model name for medical tasks | `medllama2` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `NCBI_EMAIL` | Email for PubMed/NCBI API access | `your.email@example.com` |

### Internal Constants (`src/config.py`)

| Constant | Value | Description |
|---|---|---|
| `MAX_ITERATIONS` | `3` | Maximum reflection-improvement cycles |
| `BYPASS_REFLECTION` | `True` | Skip reflection agent (useful for faster testing) |

---

## 📡 API Reference

### Stateless Query

#### `POST /api/query`

Send a one-off medical query without session tracking.

**Request:**
```json
{ "userQuery": "What are the latest treatments for Type 2 Diabetes?" }
```

**Response:**
```json
{
  "finalResponse": "Comprehensive medical report...",
  "isSimpleQuery": false,
  "qualityPassed": true
}
```

---

### Session Management

#### `POST /api/sessions`

Create a new conversation session.

**Response:**
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "createdAt": "2026-02-14T01:50:00Z",
  "message": "Session created successfully"
}
```

#### `GET /api/sessions`

List all active sessions.

**Response:**
```json
{
  "sessions": [
    {
      "sessionId": "550e8400-...",
      "createdAt": "2026-02-14T01:50:00Z",
      "updatedAt": "2026-02-14T01:55:00Z",
      "messageCount": 3
    }
  ],
  "total": 1
}
```

#### `GET /api/sessions/{session_id}`

Get session details and full conversation history.

**Response:**
```json
{
  "sessionId": "550e8400-...",
  "createdAt": "2026-02-14T01:50:00Z",
  "updatedAt": "2026-02-14T01:55:00Z",
  "history": [
    {
      "query": "What is hypertension?",
      "response": "Hypertension is...",
      "isSimple": true,
      "timestamp": "2026-02-14T01:51:00Z"
    }
  ]
}
```

#### `DELETE /api/sessions/{session_id}`

Delete a session and its conversation history.

**Response:**
```json
{ "message": "Session deleted successfully" }
```

#### `POST /api/sessions/{session_id}/query`

Submit a query within a session. Conversation history is passed to the agents for context.

**Request:**
```json
{ "userQuery": "What are the treatment options?" }
```

**Response:**
```json
{
  "sessionId": "550e8400-...",
  "finalResponse": "Based on our previous discussion...",
  "isSimpleQuery": false,
  "qualityPassed": true,
  "historyLength": 2
}
```

---

## 💡 Example Queries

| Query Type | Example |
|---|---|
| **Simple** | *"What is hypertension?"* |
| **Complex (Clinical)** | *"Compare the efficacy of SGLT2 inhibitors versus GLP-1 receptor agonists in patients with Type 2 Diabetes and cardiovascular risk."* |
| **Complex (Research)** | *"What are the latest FDA-approved treatments for Alzheimer's disease?"* |
| **Complex (Diagnostic)** | *"What are the treatment options for Type 2 Diabetes with comorbid hypertension?"* |
| **Complex (Nutrition)** | *"Explain how a low-carb diet affects weight management in PCOS."* |
| **Complex (Guidelines)** | *"Give me the current guidelines for pediatric fever management."* |

---

## 🗺 Roadmap

- [ ] **Enhanced RAG Pipeline** — Improved PubMed integration with persistent vector storage
- [ ] **Frontend Interface** — React-based UI with real-time agent progress visualization
- [ ] **Streaming Tokens** — Real-time token streaming for all agent outputs
- [ ] **Confidence Scoring** — Agent-level confidence metrics for each claim
- [ ] **Hallucination Detection** — Secondary validation via specialized fact-checking LLM
- [ ] **Plugin Architecture** — Modular plugins for scheduling, WHO/NIH guideline lookups, etc.
- [ ] **Multi-language Support** — Responses in multiple languages based on user preference

---

## 📜 License & Disclaimer

### License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software subject to the license terms.

### Disclaimer

1. **Not Medical Advice** — All information provided by this system is for **research and educational purposes only**. It is *not* a substitute for professional medical advice, diagnosis, or treatment.
2. **No Warranty** — The software is provided "as is", without warranty of any kind. Use it at your own risk.
3. **Responsibility** — By using this project, you acknowledge that the authors and contributors are not responsible for any decision you make based on the system's output. Always verify critical health information with multiple reputable sources.

---

### 🙏 Acknowledgments

Thank you for checking out the Medical Research Assistant! If you find bugs, have suggestions, or want to contribute, feel free to open an issue or submit a pull request. Happy exploring! 🚀
