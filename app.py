from fastapi import FastAPI, Query
import chromadb
import ollama

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

# This is the line that points to your Mac!
# ollama_client = ollama.Client(host="http://host.docker.internal:11434")
ollama_client = ollama.Client(host="http://localhost:11434")

@app.post("/query")
def query(q: str = Query(..., description="Query the RAG API")):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""
    
    # Using ollama_client instead of ollama!
    answer = ollama_client.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )
    return {"answer": answer["response"]}