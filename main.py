import os
from typing import Dict, Any, List
import chromadb
from tinfoil import TinfoilAI
import argparse

from embedding import chunk_messages
from embedding.db import get_embedding_collection, add_chunks_to_chroma, query_messages
from embedding.parsers import parse_imessage, parse_signal
from server import run_server

TINFOIL_API_KEY = os.getenv("TINFOIL_API_KEY")

client = TinfoilAI(
    enclave="llama3-3-70b.model.tinfoil.sh",
    repo="tinfoilsh/confidential-llama3-3-70b",
    api_key=TINFOIL_API_KEY,
)

def create_chat_response(query_results: Dict[str, Any], question: str):
    formatted_contexts = []
    for doc, meta in zip(query_results['documents'], query_results['metadatas']):
        formatted_contexts.append(doc)
    
    context = "\n\n".join(formatted_contexts)

    prompt = f"""
You are analyzing text messages from the user.
Based on the following excerpts, please answer this question: {question}

EXCERPTS:
{context}

Please provide a concise answer based only on the information in these excerpts.
Think critically and give the user a rough, unfiltered, and brutally honest answer to the question. Be concise.
Do not include anything about sex, drugs, crime, or anything that is not related to the question. Do not include any names of people.
"""

    stream = client.chat.completions.create(
        model="llama3-3-70b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing text messages."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    return stream

def respond(query_results: Dict[str, Any], question: str) -> None:
    stream = create_chat_response(query_results, question)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def interactive_query(collection: chromadb.Collection, print_excerpts: bool = False):
    while True:
        question = input("🧠 > ")
        if question.lower() in ['exit', 'quit']:
            break

        results = query_messages(collection, question)
        respond(results, question)

        if print_excerpts:
            print("\nBased on these message excerpts:")
            for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
                print(f"\n--- Excerpt {i} ({meta['start_time']} to {meta['end_time']}) ---")
                print(doc[:300] + "..." if len(doc) > 300 else doc)
            print("---" + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Message RAG Pipeline")
    parser.add_argument('--file', type=str, help='Path to the text message file')
    parser.add_argument('--format', type=str, choices=['imessage', 'signal'], help='Format of the input file (imessage or signal)')
    parser.add_argument('--excerpts', action='store_true', help='Print excerpts')
    parser.add_argument('--db', type=str, required=True, help='Path to the ChromaDB directory')
    parser.add_argument('--listen', type=int, default=0, help='Port to run the server on (default: none)')
    args = parser.parse_args()

    collection = get_embedding_collection(args.db, TINFOIL_API_KEY)

    if args.listen > 0:
        run_server(args.listen, collection, create_chat_response)
    elif args.file:
        if args.format == "":
            raise ValueError("Format is required")
        elif args.format == "imessage":
            messages = parse_imessage(args.file)
        elif args.format == "signal":
            messages = parse_signal(args.file)
        else:
            raise ValueError(f"Invalid format: {args.format}")
        print(f"Parsed {len(messages)} messages")

        chunks = chunk_messages(messages)
        print(f"Created {len(chunks)} chunks")

        add_chunks_to_chroma(collection, chunks)
        print("All chunks added to ChromaDB")

        print(f"Successfully processed {len(messages)} messages into {len(chunks)} chunks")
    else:
        interactive_query(collection, print_excerpts=args.excerpts)
