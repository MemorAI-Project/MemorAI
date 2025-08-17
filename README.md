<<<<<<< HEAD
# MemorAI v1.0 - Semantic Memory Assistant
=======
# 🧠 MemorAI v1.1 – Semantic Memory Chatbot
>>>>>>> main

MemorAI is an **AI-powered conversational agent with semantic memory**.
It uses embeddings, FAISS vector search, and summarization models to store, retrieve, and merge past conversation history, allowing the chatbot to **remember context over long interactions**.

---

## 🚀 Features

- **Semantic Memory**: Stores conversations in vector embeddings using [Sentence Transformers](https://www.sbert.net/).
- **Vector Search with FAISS**: Finds the most relevant past context for each user query.
- **Summarization Pipeline**: Extractive + abstractive summarization for compact memory storage.
- **Memory Management**:

  - Merges similar memories automatically.
  - Caps memory size and prunes older entries.
  - Persists memory in `memory.json`.

- **Streaming Chat**: Integrates with a local LLM server (`http://127.0.0.1:1234/v1/chat/completions`) for streaming responses.

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/MemorAI.git
cd MemorAI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** (suggested):

```
faiss-cpu
numpy
torch
requests
sentence-transformers
transformers
```

---

## ⚙️ Configuration

Edit these variables at the top of the script if needed:

```python
API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # Your local LLM API
MODEL_NAME = "deepseek-r1-distill-qwen-7b"             # Model name for chat API
MAX_MEMORIES = 100                                     # Max number of stored memories
SIMILARITY_THRESHOLD = 0.7                             # Memory merge threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY_FILE = "memory.json"
```

---

## ▶️ Usage

Run the chatbot:

```bash
python MemorAI.py
```

You’ll see:

```
MemorAI - Embedding-Only Semantic Memory Version
Type 'exit' to quit or '/memory' to view memory
```

### Commands

- **Normal input** → Chat with the AI.
- **/memory** → Display stored memory entries.
- **exit** → Quit the program.

---

## 🛠️ How It Works

1. **User sends input**
2. AI searches FAISS for relevant past context.
3. Context is injected into the system prompt.
4. LLM generates a response (streaming).
5. Conversation is summarized and stored in memory.
6. Similar entries are merged; old ones pruned.

---

## 📂 Project Structure

```
MemorAI/
│── MemorAI.py       # Main script
│── memory.json      # Persistent memory store
│── requirements.txt # Dependencies
│── README.md        # Documentation
```

---

## 📝 Example

```
You: hi
AI: Hello! How can I help you today?

You: /memory
[2025-08-10T14:49:09] User greeted the assistant with "hi" and received a friendly reply.
```
