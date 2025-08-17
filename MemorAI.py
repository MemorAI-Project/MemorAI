import os
import json
import faiss
import numpy as np
import torch
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging as transformers_logging
transformers_logging.set_verbosity_error()  # Suppress all transformers warnings

# ---------------- CONFIG ----------------
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "deepseek-r1-distill-qwen-7b"
MAX_MEMORIES = 100
SIMILARITY_THRESHOLD = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY_FILE = "memory.json"

# ---------------- MEMORY CLASS ----------------
class MemoryManager:
    def __init__(self, embed_model, file_path=MEMORY_FILE):
        self.embed_model = embed_model
        self.file_path = file_path
        self.memories = []
        self.faiss_index = None
        # sentence-transformers provides this helper
        try:
            self.embedding_dim = int(self.embed_model.get_sentence_embedding_dimension())
        except Exception:
            # fallback to known dim for all-MiniLM-L6-v2
            self.embedding_dim = 384
        self._load_memory()

    def _load_memory(self):
        """Load memory from disk, handle empty/corrupt files and migrate simple formats."""
        try:
            if os.path.exists(self.file_path):
                if os.path.getsize(self.file_path) > 0:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    # handle older formats: if file is list of strings, convert
                    migrated = []
                    for item in raw:
                        if isinstance(item, str):
                            migrated.append({
                                "timestamp": datetime.now().isoformat(),
                                "content": item,
                                "embedding": self._safe_embed_to_list(item)
                            })
                        elif isinstance(item, dict) and "content" in item:
                            # if embedding exists but as list, keep; if missing, compute
                            if "embedding" not in item or not item["embedding"]:
                                item["embedding"] = self._safe_embed_to_list(item["content"])
                            migrated.append(item)
                        else:
                            # unknown item -> skip
                            continue
                    self.memories = migrated
                else:
                    # empty file -> start with empty list
                    self.memories = []
            else:
                # file doesn't exist
                self.memories = []
        except json.JSONDecodeError:
            print("Warning: memory file is corrupted or empty — starting with blank memory.")
            self.memories = []
        except Exception as e:
            print(f"Warning: unexpected error loading memory: {e}")
            self.memories = []

        # ensure file exists as a valid JSON file
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        self._rebuild_faiss_index()

    def _save_memory(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save memory: {e}")

    def _rebuild_faiss_index(self):
        # create fresh index and add existing embeddings (if any)
        try:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            if self.memories:
                embeddings = [m["embedding"] for m in self.memories]
                arr = np.array(embeddings, dtype="float32")
                if arr.size > 0:
                    self.faiss_index.add(arr)
        except Exception as e:
            print(f"[Error] Rebuilding FAISS index: {e}")
            # if rebuilding fails, set index to empty index with correct dim
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def _safe_embed_to_list(self, text):
        """Return embedding as Python list (float) for JSON storage."""
        try:
            emb = self.embed_model.encode(text)
            # ensure numpy array
            emb_arr = np.array(emb, dtype="float32").flatten()
            return emb_arr.tolist()
        except Exception as e:
            print(f"[Warning] embedding failed: {e}")
            # fallback to zeros
            return np.zeros(self.embedding_dim, dtype="float32").tolist()

    def embed_text(self, text):
        """Return numpy float32 1-D embedding for use with FAISS/search."""
        emb = self.embed_model.encode(text)
        return np.array(emb, dtype="float32").flatten()

    def search(self, query, top_k=3):
        """Return top-k memory items whose distance passes the threshold check."""
        if not self.memories or self.faiss_index is None:
            return []
        query_emb = np.array([self.embed_text(query)], dtype="float32")
        try:
            distances, indices = self.faiss_index.search(query_emb, top_k)
        except Exception as e:
            print(f"[Error] FAISS search failed: {e}")
            return []
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                # NOTE: distances are L2. The numeric threshold semantics depend on your embeddings.
                # Here we reuse the pattern: consider a match if dist < (1 - SIMILARITY_THRESHOLD)
                # This mirrors previous usage; adjust if you want cosine similarity instead.
                if dist < (1.0 - SIMILARITY_THRESHOLD):
                    results.append(self.memories[idx])
        return results

    def add_or_merge_memory(self, summary):
        embedding = self.embed_text(summary)
        query_emb = np.array([embedding], dtype="float32")
        if self.memories and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                distances, indices = self.faiss_index.search(query_emb, 1)
                if distances[0][0] < (SIMILARITY_THRESHOLD):
                    # merge: update existing entry content and timestamp
                    idx = int(indices[0][0])
                    self.memories[idx]["content"] = summary
                    self.memories[idx]["timestamp"] = datetime.now().isoformat()
                    self.memories[idx]["embedding"] = embedding.tolist()
                else:
                    self.memories.append({
                        "timestamp": datetime.now().isoformat(),
                        "content": summary,
                        "embedding": embedding.tolist()
                    })
            except Exception as e:
                # if search fails for some reason, append
                print(f"[Warning] FAISS search failed during add/merge: {e}")
                self.memories.append({
                    "timestamp": datetime.now().isoformat(),
                    "content": summary,
                    "embedding": embedding.tolist()
                })
        else:
            # no existing memories -> append
            self.memories.append({
                "timestamp": datetime.now().isoformat(),
                "content": summary,
                "embedding": embedding.tolist()
            })

        # cap memory size (simple FIFO)
        if len(self.memories) > MAX_MEMORIES:
            self.memories = self.memories[-int(MAX_MEMORIES * 0.9):]

        self._save_memory()
        self._rebuild_faiss_index()

# ---------------- AI CLASS ----------------
class MemorAI:
    def __init__(self):
        print("MemorAI - Embedding-Only Semantic Memory Version")
        # load embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        self.memory = MemoryManager(self.embed_model, file_path=MEMORY_FILE)
        # summarizer device: 0 for cuda, -1 for cpu
        summarizer_device = 0 if DEVICE == "cuda" else -1
        self.summarizer = pipeline("summarization", model="t5-small", device=summarizer_device)

    def _generate_summary(self, text):
        if not text or not text.strip():
            return None
        try:
            extractive = self.summarizer(text, max_length=60, min_length=10, do_sample=False)[0]["summary_text"]
        except Exception as e:
            print(f"[Warning] extractive summarizer failed: {e}")
            # fallback: truncate text
            extractive = text.strip()[:400]
        prompt = f"Summarize this text in under 100 words:\n\n{extractive}"
        try:
            resp = requests.post(API_URL, json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            summary = data["choices"][0]["message"]["content"].strip()
            return summary
        except Exception as e:
            print(f"[Error] Abstractive summarization failed, using extractive fallback: {e}")
            return extractive

    def _update_memory(self, conversation_text):
        summary = self._generate_summary(conversation_text)
        if summary:
            self.memory.add_or_merge_memory(summary)

    def chat(self, user_input):
        context = self.memory.search(user_input)
        messages = []
        if context:
            context_text = "\n".join([m["content"] for m in context])
            messages.append({
        "role": "system",
        "content": (
            "You are an AI assistant with memory. "
            "Here are notes from past interactions with the user. "
            "Always use them to answer questions, especially if the user asks what you remember:\n\n"
            f"{context_text}\n\n" 
        )
    })
        messages.append({"role": "user", "content": user_input})

        reply = ""
        try:
            with requests.post(API_URL, json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.7,
                "stream": True
            }, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data_str = line[len(b"data: "):].decode("utf-8")
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                print(delta, end="", flush=True)
                                reply += delta
                        except json.JSONDecodeError:
                            continue
                print()
        except Exception as e:
            print(f"[Error] Chat failed: {e}")

        # asynchronously update memory with the short conversation
        Thread(target=self._update_memory, args=(f"User: {user_input}\nAssistant: {reply}",), daemon=True).start()

# ---------------- MAIN LOOP ---------------- #
if __name__ == "__main__":
    bot = MemorAI()
    print("Type 'exit' to quit or '/memory' to view memory\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        elif user_input.strip() == "/memory":
            for m in bot.memory.memories:
                print(f"[{m['timestamp']}] {m['content']}\n")
        else:
            print("AI: ", end="", flush=True)
            bot.chat(user_input)
