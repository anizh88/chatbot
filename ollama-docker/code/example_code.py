from ollama import Client
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = chromadb.Client()
remote_client = Client(host=f"http://localhost:11434")
collection = client.get_or_create_collection(name="articles_demo")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=['.', '\n']
)


print("Reading articles.jsonl and generating embeddings...")
with open("Kathmandpost_articles.json", "r",encoding="utf-8") as f:
    for i, line in enumerate(f):
        article = json.loads(line)
        content = article["content"]
        chunks = [c.strip() for c in splitter.split_text(content) if len(c.strip()) > 15]
        
        # FIX: Indent this block to process every chunk
        for j, chunk in enumerate(chunks):
            response = remote_client.embed(
                model="nomic-embed-text",
                input=f"search_document: {chunk}"
            )
            embedding = response["embeddings"][0]

            collection.add(
                ids=[f"article_{i}_chunk_{j}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"title": article["title"]}],
            )



print("Database built successfully!")


# query = "what are different problems provinces of nepal are facing?"
query = "karki commision getting thrid term?"
query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
results = collection.query(query_embeddings=[query_embed], n_results=1)
print(f"\nQuestion: {query}")
print(f'\n Title : {results["metadatas"][0][0]["title"]} \n {results["documents"][0][0]} ')