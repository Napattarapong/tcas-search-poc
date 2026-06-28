"""Pre-download bge-m3 model so the demo doesn't trigger a cold-start download."""
from src.vector_search import BgeM3Embedder

if __name__ == "__main__":
    print("Downloading/caching bge-m3 model...")
    BgeM3Embedder(cache_dir="data/models")
    print("Model ready.")