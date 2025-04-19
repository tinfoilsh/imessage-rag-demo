import re
import os
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tinfoil import TinfoilAI
from embeddings import get_embedding_collection, add_chunks_to_chroma, query_messages

COLLECTION_NAME = "text_messages"
CHUNK_SIZE = 10  # Number of messages to chunk together
OVERLAP = 2  # Number of messages to overlap between chunks

MODEL = "llama3-3-70b"
client = TinfoilAI(
    enclave="llama3-3-70b.model.tinfoil.sh",
    repo="tinfoilsh/confidential-llama3-3-70b",
    api_key=os.getenv("TINFOIL_API_KEY"),
)

collection = get_embedding_collection()

def parse_text_messages(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the text message file into a structured format.
    
    Args:
        file_path: Path to the text message file
        
    Returns:
        List of message dictionaries with timestamp, sender, and content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by empty lines to get message blocks
    message_blocks = re.split(r'\n\n+', content)
    messages = []
    
    for block in message_blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Extract timestamp, read status
        timestamp_line = lines[0]
        timestamp_match = re.match(r'(.*?)(\(Read.*\))?$', timestamp_line)
        
        if timestamp_match:
            timestamp_str = timestamp_match.group(1).strip()
            read_status = timestamp_match.group(2) if timestamp_match.group(2) else None
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, '%b %d, %Y %I:%M:%S %p')
            except ValueError:
                timestamp = None
                
            # Extract sender
            sender = lines[1].strip()
            
            # Extract message content
            content = '\n'.join(lines[2:]).strip()
            
            # Determine if sender is "Me" or the other person
            is_me = sender == "Me"
            other_person = sender if not is_me else None
            
            messages.append({
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'read_status': read_status,
                'sender': sender,
                'is_me': is_me,
                'other_person': other_person,
                'content': content
            })
    
    return messages


def chunk_messages(messages: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Dict[str, Any]]:
    """
    Chunk messages with overlap to maintain context.
    
    Args:
        messages: List of message dictionaries
        chunk_size: Number of messages per chunk
        overlap: Number of messages to overlap between chunks
        
    Returns:
        List of chunk dictionaries with ids and text
    """
    chunks = []
    
    for i in range(0, len(messages), chunk_size - overlap):
        chunk_messages = messages[i:i + chunk_size]
        if len(chunk_messages) < 2:  # Skip chunks that are too small
            continue
            
        # Format messages for this chunk
        formatted_messages = []
        for msg in chunk_messages:
            sender = "You" if msg['is_me'] else "Other"
            timestamp = msg['timestamp_str']
            formatted_messages.append(f"[{timestamp}] {sender}: {msg['content']}")
        
        chunk_text = "\n".join(formatted_messages)
        
        # Create a timestamp range for the chunk ID
        start_time = chunk_messages[0]['timestamp_str']
        end_time = chunk_messages[-1]['timestamp_str']
        chunk_id = f"chunk_{i}_{start_time}_{end_time}".replace(" ", "_").replace(":", "-")
        
        # Create chunk metadata
        metadata = {
            'start_time': start_time,
            'end_time': end_time,
            'message_count': len(chunk_messages),
            'senders': ','.join(set([msg['sender'] for msg in chunk_messages])),
        }
        
        chunks.append({
            'id': chunk_id,
            'text': chunk_text,
            'metadata': metadata
        })
    
    return chunks


def process_message_file(file_path: str) -> None:
    """
    Process the text message file and add to ChromaDB.
    
    Args:
        file_path: Path to the text message file
    """
    print(f"Processing file: {file_path}")
    messages = parse_text_messages(file_path)
    print(f"Parsed {len(messages)} messages")
    
    chunks = chunk_messages(messages)
    print(f"Created {len(chunks)} chunks")
    
    add_chunks_to_chroma(collection, chunks)
    print("All chunks added to ChromaDB")
    
    return len(messages), len(chunks)


def summarize_with_llm_streaming(query_results: Dict[str, Any], question: str) -> None:
    """
    Stream the LLM response as it's being generated.
    
    Args:
        query_results: Results from the query_messages function
        question: The original question
    """
    # Format context
    formatted_contexts = []
    for doc, meta in zip(query_results['documents'], query_results['metadatas']):
        formatted_contexts.append(doc)
    
    context = "\n\n".join(formatted_contexts)
    
    prompt = f"""
You are analyzing text messages from the user.
Based on the following excerpts, please answer this question: {question}

EXCERPTS:
{context}

Please provide a concise answer based only on the information in these excerpts. Think critically and give the user a rough, unfiltered, and brutally honest answer to the question.
"""

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing text messages."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    # Process the streaming response
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


def interactive_query(stream: bool = True, print_excerpts: bool = True) -> None:
    """
    Run an interactive query loop.
    
    Args:
        stream: Whether to stream the LLM response
    """
    while True:
        question = input("🧠 > ")
        if question.lower() in ['exit', 'quit']:
            break
            
        results = query_messages(collection, question)
        summarize_with_llm_streaming(results, question)

        if print_excerpts:
            print("\nBased on these message excerpts:")
            for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
                print(f"\n--- Excerpt {i} ({meta['start_time']} to {meta['end_time']}) ---")
                print(doc[:300] + "..." if len(doc) > 300 else doc)
            print("---" + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Message RAG Pipeline")
    parser.add_argument('--file', type=str, help='Path to the text message file')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming responses')
    parser.add_argument('--excerpts', action='store_true', help='Print excerpts')
    
    args = parser.parse_args()
    
    if args.file:
        msg_count, chunk_count = process_message_file(args.file)
        print(f"Successfully processed {msg_count} messages into {chunk_count} chunks")
    
    interactive_query(stream=not args.no_stream, print_excerpts=args.excerpts)
