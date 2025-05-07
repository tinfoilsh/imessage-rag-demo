from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

@dataclass
class Message:
    timestamp: datetime
    sender: str
    content: str

def chunk_messages(messages: List[Message], chunk_size: int = 10, overlap: int = 2) -> List[Dict[str, Any]]:
    """
    Chunk messages with overlap to maintain context.
    
    Args:
        messages: List of Message objects
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
            formatted_messages.append(f"[{msg.timestamp}] {msg.sender}: {msg.content}")

        chunk_text = "\n".join(formatted_messages)

        # Create a timestamp range for the chunk ID
        start_time = int(chunk_messages[0].timestamp.timestamp())
        end_time = int(chunk_messages[-1].timestamp.timestamp())
        chunk_id = f"chunk_{i}_{start_time}_{end_time}".replace(" ", "_").replace(":", "-")

        chunk = {
            'id': chunk_id,
            'text': chunk_text,
            'metadata': {
                'start_time': start_time,
                'end_time': end_time,
                'message_count': len(chunk_messages),
                'senders': ','.join(set([msg.sender for msg in chunk_messages])),
            }
        }
        chunks.append(chunk)

    return chunks
