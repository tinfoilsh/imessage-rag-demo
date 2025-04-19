# Tinfoil Secure iMessage RAG Demo

This project demonstrates how to use Tinfoil to build a secure RAG pipeline for iMessage.

## Prerequisites

- Python 3.12+
- macOS (for iMessage access)
- [imessage-exporter](https://github.com/ReagentX/imessage-exporter)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/tinfoilsh/imessage-rag-demo
cd imessage-rag-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Install imessage-exporter:
```bash
git clone https://github.com/ReagentX/imessage-exporter.git
cargo build --release
./target/release/imessage-exporter --format txt
```

## Processing and Embedding Messages

1. Process the exported messages and create embeddings:
```bash
python main.py --file texts.txt
```
