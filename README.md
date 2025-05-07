# Tinfoil Secure Text RAG

## Build

```bash
docker build -t tinfoil-chat-rag .
```

## Embedding

```bash
docker run --rm -it \
    -e TINFOIL_API_KEY=none \
    -v $(pwd)/signal-chat.json:/chat.json \
    -v $(pwd)/vdb:/db \
    tinfoil-chat-rag --db /db --format signal --file /chat.json
```

## Inference

```
docker run --rm -it \
    -e TINFOIL_API_KEY=none \
    -v $(pwd)/vdb:/db \
    tinfoil-chat-rag --db /db
```
