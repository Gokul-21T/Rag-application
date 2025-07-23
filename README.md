# 📚 RAG Chatbot App

A modular and production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**. This application allows users to upload PDFs, embed their content, and interact with a **locally hosted LLM** to ask context-aware questions based on the document content.

---

## 📑 Table of Contents

- [🚀 Features](#-features)
- [⚙️ Setup Instructions](#️-setup-instructions)
- [🧠 Usage](#-usage)
- [🏗️ Architectural Decisions](#-architectural-decisions)
- [🔍 Chunking Strategy](#-chunking-strategy)
- [📥 Retrieval Approach](#-retrieval-approach)
- [🖥️ Hardware Utilization](#️-hardware-utilization)
- [🐛 Development Challenges](#-development-challenges)
- [🧰 Troubleshooting](#-troubleshooting)
- [📁 File Structure](#-file-structure)
- [📄 License](#-license)

---

## 🚀 Features

- Upload and embed PDFs securely.
- Local LLM integration for private inference.
- Real-time chat with document-aware responses.
- Hardware-aware acceleration with GPU/CPU fallback.
- Transparent display of context chunks and metadata.
- Clean and modular codebase for scalability.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Gokul-21T/rag.git
cd rag
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Files

Model files are not included in the repository due to size restrictions. Use the following script to download the **Phi-3 Mini GGUF** model:

```bash
python rag_chatbot_app/download_phi3_model.py
```

* Models are saved in: `rag_chatbot_app/models/Phi-3-mini-4k-instruct-gguf/`
* For other models, manually place them in a new subdirectory under `rag_chatbot_app/models/`.

### 5. Run the Application

```bash
streamlit run rag_chatbot_app/app.py
```

### 6. (Optional) Clean Cache & Configure `.gitignore`

#### Remove unnecessary local folders:

```bash
rmdir /s /q __pycache__
rmdir /s /q .streamlit\cache
rmdir /s /q .streamlit\cache_data
rmdir /s /q venv
```

#### Add to `.gitignore`:

```bash
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".streamlit/cache/" >> .gitignore
echo ".streamlit/cache_data/" >> .gitignore
echo "rag_chatbot_app/models/" >> .gitignore
```

---

## 🧠 Usage

1. Open the Streamlit UI.
2. Upload a PDF document.
3. The app extracts and embeds text using a local embedding model.
4. Ask questions in the chat box.
5. The chatbot retrieves relevant chunks and answers your query using a local LLM.
6. View chat history and the source chunks used for transparency.

---

## 🏗️ Architectural Decisions

* **Framework**: Built using **Streamlit** for rapid UI development.
* **LLMs**: Supports local models (e.g., Mistral, Phi-3) loaded via Transformers or GGUF loaders.
* **Embeddings**: Generated locally using lightweight models like **MiniLM**.
* **Vector Store**: Utilizes **FAISS** or **Qdrant** for efficient similarity search.
* **Modular Codebase**: Separate modules for RAG logic, utility functions, and model handling.
* **Hardware Detection**: Automatically uses GPU if available; otherwise, defaults to CPU.

---

## 🔍 Chunking Strategy

* PDFs are parsed and chunked into fixed-length texts (e.g., 500–1000 characters/tokens).
* Overlaps of ~50–100 tokens ensure context continuity.
* Each chunk includes metadata: filename, page number, and chunk ID.
* Enables granular referencing and improved context retrieval accuracy.

---

## 📥 Retrieval Approach

1. User query is embedded via the local embedding model.
2. Top-K similar chunks are fetched from the vector store.
3. Retrieved chunks are passed as context to the LLM.
4. LLM answers strictly using the provided context.
5. Source chunks and metadata are displayed for transparency.

---

## 🖥️ Hardware Utilization

* **GPU Acceleration**: Automatically used if CUDA is available for LLM and embeddings.
* **CPU Fallback**: Functions entirely on CPU if no GPU is present.
* **UI Display**: Device status is shown to the user in the app header.

---

## 🐛 Development Challenges

| Area                        | Challenge                                         | Resolution                                                  |
| --------------------------- | ------------------------------------------------- | ----------------------------------------------------------- |
| **Large Files**             | GitHub file size limits for models/venv           | `.gitignore` and BFG Repo-Cleaner used to manage size       |
| **CUDA Issues**             | Missing/unsupported GPU drivers                   | Auto-fallback to CPU; ensure CUDA & PyTorch compatibility   |
| **Model Download Failures** | Incorrect repo or filename                        | Verified URLs and added validation in scripts               |
| **Streamlit Session Bugs**  | Unintended reruns and lost states                 | Managed with `st.session_state` effectively                 |
| **Package Conflicts**       | Dependency issues (e.g., `torch`, `transformers`) | Used isolated virtual environments with strict requirements |
| **File Access Errors**      | OS locking files                                  | Used `with open(...)` patterns to avoid locks               |

---

## 🧰 Troubleshooting Guide

| Problem                          | Cause                         | Solution                                             |
| -------------------------------- | ----------------------------- | ---------------------------------------------------- |
| `Push rejected due to file size` | Large model or venv files     | Clean with BFG & update `.gitignore`                 |
| `CUDA not available`             | No GPU or driver mismatch     | App runs on CPU; install correct drivers             |
| `Model 404 error`                | Invalid model URL or filename | Cross-check script and Hugging Face repo             |
| `Streamlit UI flickers`          | Improper rerun triggers       | Use `st.session_state` and avoid `st.rerun()` misuse |
| `Import errors`                  | Outdated/missing packages     | Recreate environment and reinstall                   |
| `Permission/File access errors`  | File is still open            | Use safe file handling via `with` statement          |

---

## 📁 File Structure

```
rag/
├── rag_chatbot_app/
│   ├── app.py                  # Main Streamlit app
│   ├── download_phi3_model.py # Phi-3 model download script
│   ├── models/                 # Folder to store downloaded models
│   ├── utils.py                # Utility functions
│   └── rag_engine.py          # Core RAG pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files and folders
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. 
