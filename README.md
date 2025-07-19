# MemorAI - Semantic Memory Assistant

## Overview

MemorAI is an advanced conversational AI assistant with long-term semantic memory capabilities. It uses a hybrid approach of extractive and abstractive summarization combined with semantic search to maintain and retrieve relevant context from past conversations.

Key features:
- Persistent memory storage in JSON format
- Semantic search using FAISS vector database
- Hybrid summarization pipeline (extractive + abstractive)
- Keyword extraction with KeyBERT
- Memory merging based on semantic similarity
- Streaming chat responses

## Technical Components

### Core Technologies
- **Language Model**: Integrates with local LLM API (default: deepseek-r1-distill-qwen-7b)
- **Embeddings**: Uses `all-MiniLM-L6-v2` from SentenceTransformers
- **Vector Database**: FAISS for efficient similarity search
- **Keyword Extraction**: KeyBERT with paraphrase-MiniLM-L3-v2
- **Summarization**: T5-small for extractive summarization + LLM for abstractive refinement

### Memory Management
- Stores conversation summaries with timestamps and keywords
- Automatically merges similar memories
- Enforces maximum memory limit with LRU-like eviction
- Semantic search for context retrieval

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU recommended
- Local LLM API endpoint (default expects http://127.0.0.1:1234/v1/chat/completions)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/memorai.git
   cd memorai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the API endpoint in `main.py` if needed:
   ```python
   API_URL = "http://your-api-endpoint/v1/chat/completions"
   MODEL_NAME = "your-model-name"
   ```

## Usage

Run the assistant:
```bash
python main.py
```

### Commands
- Regular chat: Just type your message
- View memory: `/memory`
- Exit: `exit` or `quit`

## Configuration

Modify these constants in `main.py` as needed:

```python
API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # Your LLM API endpoint
MODEL_NAME = "deepseek-r1-distill-qwen-7b"            # Model name
MEMORY_FILE = "memory.json"                           # Memory storage file
MAX_MEMORIES = 100                                    # Maximum memories to retain
SIMILARITY_THRESHOLD = 0.7                            # Similarity threshold for merging
```

## Performance Notes

- The system is optimized for GPU usage (automatically detects CUDA)
- FAISS index is kept in memory for fast retrieval
- Embeddings are cached with LRU strategy
- Memory updates happen asynchronously to avoid blocking chat

## Customization

To adapt the system:
1. Change the summarization pipeline in `MemoryManager`
2. Adjust keyword extraction parameters in `_extract_keywords()`
3. Modify the memory merging logic in `_update_memory()`

## License

[MIT License](LICENSE)
