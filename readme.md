# MedAgent: Intelligent Medical Research Assistant 🏥

MedAgent is a sophisticated AI-powered medical research assistant designed to navigate complex medical queries, retrieving and synthesizing information from trusted sources like PubMed and the broader web.

## 🌟 Key Features

*   **Multi-Agent Architecture**: Utilizes a graph of specialized AI agents (Web Search, PubMed RAG, MedILlama) orchestrated by a central planner.
*   **Real-time Streaming**: Provides token-by-token responses and live updates on agent activities.
*   **Medical Accuracy**: Cross-references findings with PubMed literature (RAG) and domain-specific knowledge.
*   **Reflection & Self-Correction**: Automatically critiques and refines answers to ensure quality.
*   **Session Management**: Persists conversation history using MongoDB.
*   **Dual Interfaces**:
    *   **Web UI**: A polished Streamlit application for interactive use.
    *   **CLI**: A robust command-line interface for developer testing and headless operation.
*   **Multilingual Support**: Capable of processing queries and generating potential responses in multiple languages (English, Spanish, French, German, Arabic, etc.), leveraging the polyglot capabilities of the underlying LLM.

## 🧠 How It Works

MedAgent operates on a **LangGraph-based state machine** that coordinates specialized agents:

1.  **Input Processing**: User queries are received via the Streamlit UI or CLI.
2.  **Evaluation Agent**: Analyzes the query complexity.
    *   *Simple queries* (e.g., "What is a fever?") are answered immediately.
    *   *Complex queries* trigger the full research workflow.
3.  **Orchestration**: The **Orchestration Agent** breaks down complex queries into sub-tasks (e.g., "Search for latest diabetes treatments", "Check side effects").
4.  **Specialized Workers**:
    *   **Web Search Agent**: Uses Tavily API to find real-time medical data and clinical trials.
    *   **MedILlama Agent**: A specialized RAG agent (simulated/fine-tuned) that provides domain-specific medical knowledge.
5.  **Compilation**: The **Compile Agent** synthesizes findings from all workers into a cohesive answer.
6.  **Reflection & Iterate**: The **Reflection Agent** critiques the answer for medical accuracy and completeness. If the quality is insufficient, the cycle repeats (up to 3 times) to refine the result.
7.  **Response**: The final verified answer is streamed to the user.


## 🚀 Getting Started

### Prerequisites

*   **Python 3.10+**
*   **MongoDB**: Run locally or use Atlas.
*   **Groq API Key**: For the LLM engine.
*   **Tavily API Key**: For web search capabilities.
*   **NCBI Email**: For PubMed API access.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/MedAgent.git
    cd MedAgent
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key
    TAVILY_API_KEY=your_tavily_api_key
    MONGODB_URI=mongodb://localhost:27017
    MONGODB_DB_NAME=medagent
    NCBI_EMAIL=your.email@example.com
    ```

## 🛠️ Usage

### Run the Web Interface (Streamlit)
The recommended way to use MedAgent.
```bash
streamlit run streamlit_app.py
```
Access the app at `http://localhost:8501`.

### Run the Backend Server
Start the FastAPI server for API access.
```bash
uvicorn src.server.app:app --reload
```
API Documentation: `http://localhost:8000/docs`

### Run the CLI
Use the command-line interface for quick queries.
```bash
python src/main.py
```

## 🧪 Testing

Run the unit tests to verify system integrity.
```bash
pytest tests/
```

## 📂 Project Structure

```
MedAgent/
├── demos/               # Demo videos
├── src/
│   ├── agents/          # Agent implementations (MedILlama, WebSearch, RAG, etc.)
│   ├── server/          # FastAPI backend application
│   ├── schemas/         # Pydantic models (API, Session, Agents)
│   ├── utils/           # Utility functions and prompts
│   ├── config.py        # Application configuration
│   ├── main.py          # CLI entry point
│   └── agent_graph.py   # LangGraph workflow definition
├── tests/               # Unit and integration tests
├── streamlit_app.py     # Streamlit frontend
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🎥 Demo

Watch the full workflow in action, including real-time reasoning and multi-agent coordination:

▶️ **[View Demo Video](demos/MedAgent_demo.mp4)**

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License

This project is licensed under the MIT License.
