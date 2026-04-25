import chromadb
client = chromadb.Client()
col = client.create_collection("test")
col.add(ids=["1"], embeddings=[[0.1, 0.2]], documents=["test"])
try:
    col.query(query_embeddings=[[]], n_results=1)
except Exception as e:
    print(f"Error type: {type(e)}")
    print(f"Error msg: {e}")
