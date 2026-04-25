import json

try:
    import urllib.request
    url = "http://localhost:11434/api/embeddings"
    data = json.dumps({"model": "nomic-embed-text", "prompt": ""}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as f:
        res = json.loads(f.read().decode("utf-8"))
        print(f"Embedding length: {len(res.get('embedding', []))}")
except Exception as e:
    print(f"Error: {e}")

