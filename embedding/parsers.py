from datetime import datetime
import re
import json
from typing import List
from . import Message

def parse_signal(file_path: str) -> List[Message]:
    messages = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                message = Message(
                    timestamp=datetime.fromisoformat(data['date']),
                    sender=data['sender'],
                    content=data['body'].strip()
                )
                messages.append(message)
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line}")
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    return messages

def parse_imessage(file_path: str) -> List[Message]:
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
            try:
                timestamp = datetime.strptime(
                    timestamp_match.group(1).strip(), 
                    '%b %d, %Y %I:%M:%S %p',
                )
            except ValueError:
                timestamp = None

            msg = Message(
                timestamp=timestamp,
                sender=lines[1].strip(),
                content='\n'.join(lines[2:]).strip()
            )
            messages.append(msg)
    
    return messages
