import json
import os
import re
import requests
import threading
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import torch
import faiss
from transformers import pipeline,logging as transformers_logging
transformers_logging.set_verbosity_error()  # Suppress all transformers warnings

# Configuration
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "deepseek-r1-distill-qwen-7b"
MEMORY_FILE = "memory.json"
MAX_MEMORIES = 100  # Maximum number of memories to retain
SIMILARITY_THRESHOLD = 0.7  # For memory merging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MemoryManager:
    def __init__(self):
        self.memory = self._load_memory()
        self.kw_model = KeyBERT(model="paraphrase-MiniLM-L3-v2")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        self._init_faiss_index()
        
        self.summarizer = pipeline(
            "summarization",
            model="t5-small",
            device=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else None
        )
    
    def _init_faiss_index(self):
        self.faiss_index = faiss.IndexFlatL2(384)  # Dimension for all-MiniLM-L6-v2
        if self.memory["summaries"]:
            embeddings = [self._get_embedding(mem["content"]) for mem in self.memory["summaries"]]
            self.faiss_index.add(embeddings)
    
    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text)
    
    def _load_memory(self) -> Dict:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        return {"summaries": []}
      
    def save(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def clean_for_memory(self, text: str) -> str:
        """Remove <think> tags and sanitize input"""
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return re.sub(r'[^\w\s.,?;:\'"!@#$%^&*()\-+=]', '', clean_text)[:1000]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction with MMR diversity"""
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                use_mmr=True,
                diversity=0.5,
                top_n=8
            )
            return [kw[0] for kw in keywords if kw[1] > 0.2]  # Confidence threshold
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []

    def _summarize_chunk(self, chunk: str) -> str:
        """Helper for parallel summarization"""
        return self.summarizer(
            chunk,
            max_length=150,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
    
    def generate_summary(self, conversation: List[Dict]) -> str:
        """Hybrid summarization pipeline with parallel processing"""
        try:
            # 1. Create clean conversation text
            clean_convo = '\n'.join(
                f"{msg['role']}: {self.clean_for_memory(msg['content'])}" 
                for msg in conversation
            )
            
            # 2. Extract keywords
            keywords = self._extract_keywords(clean_convo)
            
            # 3. Parallel extractive summarization
            chunks = [clean_convo[i:i+1000] for i in range(0, len(clean_convo), 1000)]
            
            with ThreadPoolExecutor() as executor:
                extractive_summaries = list(executor.map(self._summarize_chunk, chunks))
            
            intermediate_summary = ' '.join(extractive_summaries)
            
            # 4. Abstractive refinement with main LLM
            prompt = (
                "Create a concise summary using these keywords: "
                f"{', '.join(keywords)}\n\n"
                f"Text: {intermediate_summary}\n\n"
                "Summary should be under 100 words:"
            )
            
            response = requests.post(API_URL, json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            })
            
            if response.status_code != 200:
                raise ConnectionError(f"API failed: {response.text}")
                
            response_data = response.json()
            if "choices" not in response_data:
                raise ValueError("Invalid API response structure")
            
            return self.clean_for_memory(
                response_data["choices"][0]["message"]["content"].strip()
            )
            
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback: Concatenate first/last parts
            return f"{conversation[0]['content'][:50]}...{conversation[-1]['content'][-50:]}"

class MemorAI:
    def __init__(self):
        self.memory = MemoryManager()
        self.chat_history = []
    
    def get_context(self, message: str) -> str:
        """Semantic context retrieval using FAISS"""
        if not self.memory.memory["summaries"]:
            return ""
            
        query_embedding = self.memory._get_embedding(message)
        _, indices = self.memory.faiss_index.search(
            query_embedding.reshape(1, -1), 
            k=3  # Top 3 most similar
        )
        
        context = []
        for idx in indices[0]:
            if idx < len(self.memory.memory["summaries"]):
                context.append(self.memory.memory["summaries"][idx]["content"])
        
        return '\n\n'.join(context) if context else ""
    
    def chat(self, message: str):
        """Process user input with async memory updates"""
        if message.lower() == "/memory":
            self.show_memory_preview()
            return
            
        # Get context using semantic search
        context = self.get_context(message)
        
        # Prepare messages
        messages = [{"role": "user", "content": message}]
        if context:
            messages.insert(0, {"role": "system", "content": context})
        
        # Stream response
        print("AI: ", end="", flush=True)
        full_response = ""
        
        try:
            with requests.post(API_URL, json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.7,
                "stream": True
            }, stream=True) as resp:
                
                for line in resp.iter_lines():
                    if line.startswith(b"data: "):
                        data = line[6:]
                        if data.strip() == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            print(content, end="", flush=True)
                            full_response += content
                        except json.JSONDecodeError:
                            continue
                print("\n")
                
        except Exception as e:
            error_msg = f"\n[Error] {e}"
            print(error_msg)
            full_response = error_msg
        
        # Async memory update
        threading.Thread(
            target=self._update_memory,
            args=(message, full_response),
            daemon=True
        ).start()
    
    def _update_memory(self, user_msg: str, ai_response: str):
        """Optimized memory update with semantic merging"""
        conversation = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ai_response}
        ]
        
        # Generate summary
        summary = self.memory.generate_summary(conversation)
        summary_embedding = self.memory._get_embedding(summary)
        
        # Find similar existing memory
        existing_idx = None
        if self.memory.memory["summaries"]:
            distances, indices = self.memory.faiss_index.search(
                summary_embedding.reshape(1, -1), 
                k=1
            )
            if distances[0][0] < (1 - SIMILARITY_THRESHOLD):
                existing_idx = indices[0][0]
        
        keywords = self.memory._extract_keywords(summary)
        
        if existing_idx is not None and existing_idx < len(self.memory.memory["summaries"]):
            # Merge with existing memory
            old = self.memory.memory["summaries"][existing_idx]
            merged_text = f"{old['content']}\n{summary}"
            
            merged_summary = self.memory.generate_summary([
                {"role": "system", "content": merged_text}
            ])
            
            # Update keywords (deduplicate and limit)
            merged_keywords = list(set(old["keywords"] + keywords))
            if len(merged_keywords) > 10:
                keyword_scores = {
                    kw: (old["keywords"].count(kw) * 2 + 
                         (1 if kw.lower() in merged_summary.lower() else 0)
                    for kw in merged_keywords)
                }
                merged_keywords = sorted(
                    keyword_scores.keys(),
                    key=lambda x: keyword_scores[x],
                    reverse=True
                )[:10]
            
            # Update memory entry
            self.memory.memory["summaries"][existing_idx] = {
                "timestamp": datetime.now().isoformat(),
                "content": merged_summary,
                "keywords": merged_keywords,
                "embedding": self.memory._get_embedding(merged_summary).tolist()
            }
            
            # Update FAISS index
            self.memory.faiss_index.remove_ids(existing_idx)
            self.memory.faiss_index.add(
                self.memory._get_embedding(merged_summary).reshape(1, -1)
            )
            
        else:
            # Add new memory
            new_memory = {
                "timestamp": datetime.now().isoformat(),
                "content": summary,
                "keywords": keywords[:10],  # Enforce limit
                "embedding": summary_embedding.tolist()
            }
            
            self.memory.memory["summaries"].append(new_memory)
            
            # Update FAISS index
            self.memory.faiss_index.add(summary_embedding.reshape(1, -1))
            
            # Enforce memory limit
            if len(self.memory.memory["summaries"]) > MAX_MEMORIES:
                self.memory.memory["summaries"] = sorted(
                    self.memory.memory["summaries"],
                    key=lambda x: x["timestamp"],
                    reverse=True
                )[:int(MAX_MEMORIES * 0.9)]
                self._rebuild_faiss_index()
        
        self.memory.save()
    
    def _rebuild_faiss_index(self):
        """Recreate FAISS index after major changes"""
        self.memory.faiss_index.reset()
        embeddings = [
            self.memory._get_embedding(mem["content"]) 
            for mem in self.memory.memory["summaries"]
        ]
        if embeddings:
            self.memory.faiss_index.add(embeddings)
    
    def show_memory_preview(self):
        """Display memory overview to user"""
        print("\n=== Memory Preview ===")
        for i, mem in enumerate(self.memory.memory["summaries"][:5]):
            print(f"{i+1}. {mem['content'][:80]}... (Keywords: {', '.join(mem['keywords'][:3])})")
        print(f"\nTotal memories: {len(self.memory.memory['summaries'])}/{MAX_MEMORIES}")
        print("=====================\n")

def main():
    print("MemorAI - Enhanced Semantic Memory Version")
    print("Type 'exit' to quit or '/memory' to view memory\n")
    
    ai = MemorAI()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            ai.chat(user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}. Continuing...")

if __name__ == "__main__":
    main()