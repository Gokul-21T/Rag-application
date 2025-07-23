# RAG Chatbot App

This is a Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit. It allows you to upload PDFs, embed their content, and chat with a local LLM using context retrieved from your documents.

---

## Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone https://github.com/Gokul-21T/rag.git
   cd rag
   ```

2. **Create and Activate a Virtual Environment**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Requirements**
   ```sh
   pip install -r requirements.txt
   ```

4. **Download Model Files**
   - Model files are **NOT** included in the repository. Use the provided script to download them:
   ```sh
   python rag_chatbot_app/download_phi3_model.py
   ```
   - This will download the model to `rag_chatbot_app/models/Phi-3-mini-4k-instruct-gguf/`.
   - For other models, place them in the appropriate subdirectory under `rag_chatbot_app/models/`.

5. **Run the Streamlit App**
   ```sh
   streamlit run rag_chatbot_app/app.py
   ```

---

## Usage
- Upload a PDF using the web interface.
- The app will extract and embed the document's text into a vector store.
- Ask questions in the chat box; the app retrieves relevant context and generates an answer using the local LLM.
- The chat history and context chunks are displayed for transparency.

---

## Architectural Decisions
- **Streamlit** is used for rapid prototyping and interactive UI.
- **Local LLMs** (e.g., Mistral, Phi-3) are loaded using Hugging Face Transformers and/or GGUF-compatible loaders.
- **Embeddings** are generated using a local embedding model (e.g., MiniLM) for privacy and speed.
- **Vector Store**: The app uses an in-memory or local vector database (e.g., Qdrant or FAISS) for fast similarity search.
- **Chunking**: Documents are split into overlapping text chunks to maximize retrieval accuracy and context coverage.
- **Hardware Awareness**: The app detects CUDA availability and uses GPU acceleration for both LLM and embedding models when possible, falling back to CPU if necessary.
- **Separation of Concerns**: The codebase is modular, with separate files for utility functions, RAG engine logic, and model download scripts.

---

## Chunking Strategy
- PDFs are parsed and split into text chunks of a fixed size (e.g., 500-1000 characters or N tokens), with overlap (e.g., 50-100 tokens) to preserve context across chunk boundaries.
- Each chunk is associated with metadata: filename, page number, and chunk ID.
- This strategy ensures that answers can reference precise locations in the source document and that context is not lost at chunk edges.

---

## Retrieval Approach
- When a user asks a question, the app computes the embedding of the query.
- It performs a similarity search in the vector store to retrieve the top-k most relevant chunks.
- The retrieved context is injected into the LLM prompt, instructing the model to answer **only** using the provided context.
- The app displays both the answer and the actual context chunks used, along with their metadata.

---

## Hardware Usage
- **GPU Acceleration**: If CUDA is available, both the LLM and embedding model run on the GPU for faster inference and embedding generation.
- **CPU Fallback**: If no GPU is detected, the app runs on CPU, which is slower but still functional.
- The app displays the detected hardware status at the top of the UI for user awareness.

---

## Observations
- Local RAG with GPU acceleration provides fast, private, and cost-effective document QA.
- Chunk overlap and metadata display improve answer traceability and user trust.
- Large models require significant RAM/VRAM; quantized models (e.g., GGUF) are recommended for consumer hardware.
- Ignoring model files in git and using download scripts keeps the repository lightweight and maintainable.

---

## File Structure
- `app.py` - Main Streamlit app
- `download_phi3_model.py` - Script to download Phi-3 GGUF model
- `models/` - Directory for model files (ignored by git)
- `utils.py`, `rag_engine.py` - Supporting code

---

## License
MIT 