import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
import chromadb

from embedding.db import query_messages

def create_app(collection, create_chat_response_fn):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            return {"error": "No messages provided"}

        # Get the last user message
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if not user_messages:
            return {"error": "No user message found"}

        question = user_messages[-1]["content"]
        results = query_messages(collection, question)
        stream = create_chat_response_fn(results, question)

        async def generate():
            try:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                        # Force flush the response
                        await asyncio.sleep(0)
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Error in stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )

    return app

def run_server(port: int, collection: chromadb.Collection, create_chat_response_fn):
    app = create_app(collection, create_chat_response_fn)
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port) 
