import os
from Bio import Entrez
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.schemas.state import GraphState
from src.config import logger

async def pubmed_rag_agent(state: GraphState) -> dict:
    """
    Searches PubMed and uses RAG to retrieve detailed medical information.
    Uses Biopython for fetching and Chroma for vector search.
    """
    logger.info("📚 PubMed RAG Agent Started")
    
    # Check if RAG is required
    required = state.get("requiredAgents", {}).get("rag", False)
    if not required:
        # Check tasks anyway just in case
        tasks = state.get("tasks", {}).get("RAG", [])
        if not tasks:
            return {"ragResponse": ""}

    tasks = state.get("tasks", {}).get("RAG", [])
    if not tasks and state.get("userQuery"):
       # Fallback to user query if RAG was requested but no specific tasks derived
       queries = [state["userQuery"]]
    else:
       queries = [t.query if hasattr(t, 'query') else str(t) for t in tasks]

    if not queries:
        return {"ragResponse": ""}

    # 1. Search PubMed via Biopython
    Entrez.email = os.getenv("NCBI_EMAIL", "your.email@example.com")
    
    all_abstracts = []
    
    for query in queries:
        logger.info(f"Searching PubMed for: {query}")
        try:
            # Search for IDs
            handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]
            
            if not id_list:
                continue
                
            # Fetch details
            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            papers = Entrez.read(handle)
            handle.close()
            
            for paper in papers['PubmedArticle']:
                try:
                    article = paper['MedlineCitation']['Article']
                    title = article.get('ArticleTitle', 'No Title')
                    abstract_list = article.get('Abstract', {}).get('AbstractText', [])
                    abstract = " ".join(abstract_list) if abstract_list else "No Abstract"
                    pmid = paper['MedlineCitation']['PMID']
                    
                    if abstract != "No Abstract":
                        all_abstracts.append(Document(
                            page_content=abstract,
                            metadata={"source": f"PMID:{pmid}", "title": title}
                        ))
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            continue

    if not all_abstracts:
        return {"ragResponse": "No relevant PubMed articles found."}

    # 2. Vector Store (In-memory Chroma for this session)
    logger.info("Embedding and retrieving...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_abstracts)
    
    # Store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Retrieve based on original queries
    combined_context = ""
    for query in queries:
        docs = retriever.invoke(query)
        for doc in docs:
            combined_context += f"Source: {doc.metadata['source']} - {doc.metadata['title']}\n{doc.page_content}\n\n"
            
    logger.info("✅ PubMed RAG Completed")
    return {
        "ragResponse": combined_context
    }
