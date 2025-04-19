import os
import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "text_messages"
PERSIST_DIRECTORY = "./chroma_db"

def get_embedding_collection():
    """
    Get or create a ChromaDB collection with the OpenAI embedding function.
    
    Returns:
        ChromaDB collection
    """
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key="tinfoil",
        api_base="http://localhost:11434/v1",
        model_name="nomic-embed-text"
    )

    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )
    except:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )
        print(f"Created new collection '{COLLECTION_NAME}'.")
    
    return collection

def add_chunks_to_chroma(collection, chunks):
    """
    Add message chunks to ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of chunk dictionaries
    """
    # Prepare data for ChromaDB
    ids = [chunk['id'] for chunk in chunks]
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Add to collection in batches
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        print(f"Adding {len(batch_ids)} chunks to ChromaDB (batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1})")
        
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

def query_messages(collection, question: str, n_results: int = 5):
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
