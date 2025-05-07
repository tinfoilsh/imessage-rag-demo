import os
from typing import Any, Dict
import chromadb
from embedding.tinfoil_embedding import TinfoilAIEmbeddingFunction


def get_embedding_collection(chroma_dir: str, tinfoil_api_key: str) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection with the OpenAI embedding function.
    """

    os.makedirs(chroma_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    tinfoil_ef = TinfoilAIEmbeddingFunction(
        api_key=tinfoil_api_key,
        enclave="nomic-embed-text.model.tinfoil.sh",
        repo="tinfoilsh/confidential-nomic-embed-text",
        model_name="nomic-embed-text"
    )

    COLLECTION_NAME = "text_messages"

    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=tinfoil_ef
        )
    except:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=tinfoil_ef
        )
        print(f"Created new collection {COLLECTION_NAME}")

    return collection

def add_chunks_to_chroma(collection: chromadb.Collection, chunks):
    """
    Add message chunks to ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of chunk dictionaries
    """

    ids = [chunk['id'] for chunk in chunks]
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Add to collection in batches
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        print(f"Adding {len(batch_ids)} chunks (batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1})")
        
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

def query_messages(collection, question: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Query the message database with a natural language question.
    
    Args:
        collection: ChromaDB collection
        question: Natural language question
        n_results: Number of results to return
        
    Returns:
        List of result dictionaries
    """
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    return {
        'question': question,
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0],
        'ids': results['ids'][0],
        'distances': results['distances'][0]
    }
