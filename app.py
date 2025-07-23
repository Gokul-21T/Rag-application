import streamlit as st
import tempfile
import os
from utils import extract_text_from_pdf
from rag_engine import SimpleRAG
import json
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
import torch
import concurrent.futures

st.title("RAG Chatbot App")

# Show CUDA status
cuda_status = torch.cuda.is_available()
llm_device = None
embedding_device = None
if cuda_status:
    llm_device = torch.cuda.get_device_name(0)
    try:
        # Try to get embedding model device if already loaded
        if 'rag' in st.session_state:
            embedding_model = st.session_state['rag'].embedding_model
            embedding_device = next(embedding_model.parameters()).device
    except Exception:
        embedding_device = 'Unknown'
    st.info(f"CUDA is available. LLM device: {llm_device}. Embedding model device: {embedding_device}")
else:
    st.warning("CUDA is NOT available. Running on CPU. This will be much slower.")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Initialize RAG engine (connects to local Qdrant)
if 'rag' not in st.session_state:
    st.session_state['rag'] = SimpleRAG()

# Load the local GPTQ model only once
if 'llm' not in st.session_state:
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'mistral-7b-instruct-v0.2-GPTQ-4bit-32g-actorder_True')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device_map="auto",  # Uses CUDA if available
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        revision=None,
    )
    st.session_state['llm'] = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2
    )

# PDF upload and processing
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name
    st.success("PDF uploaded. Extracting and embedding...")
    # Extract and embed
    chunks = extract_text_from_pdf(tmp_pdf_path)
    filename = uploaded_file.name if uploaded_file.name else 'document'
    st.session_state['rag'].embed_chunks(chunks, filename=filename)
    st.success("PDF processed and embedded!")
    os.remove(tmp_pdf_path)

# Chat interface
st.header("Chat")
for chat in st.session_state['chat_history']:
    st.write(f"**User:** {chat['question']}")
    st.write(f"**Bot:** {chat['answer']}")
    # Show context metadata and text if present
    if 'context_metadata' in chat:
        st.markdown("**Context Chunks Used:**")
        for meta in chat['context_metadata']:
            with st.expander(f"File: {meta.get('filename', 'N/A')} | Page: {meta.get('page_number', 'N/A')} | Chunk ID: {meta.get('chunk_id', 'N/A')}"):
                st.write(meta.get('text', 'No text found in chunk.'))

user_input = st.text_input("Ask a question:")
if st.button("Send") and user_input:
    # Retrieve context from Qdrant (top 1 chunk for focus)
    context_chunks = st.session_state['rag'].query(user_input, top_k=1)
    context = "\n".join([chunk['text'] for chunk in context_chunks])
    # Improved prompt for concise, context-only answers
    prompt = (
        f"<s>[INST] Context:\n{context}\n\n"
        f"Question: {user_input}\n"
        "Answer concisely and directly, using only the information from the context above. "
        "Do not repeat the context. If the answer is not in the context, say 'Not found in document.' [/INST]"
    )
    llm = st.session_state['llm']
    with st.spinner("Generating answer..."):
        result = llm(prompt)
        answer = result[0]['generated_text'].split('Answer:')[-1].strip()
        # Post-process: take only the first paragraph
        answer = answer.split('\n\n')[0].split('\n')[0].strip()
    st.session_state['chat_history'].append({
        "question": user_input,
        "answer": answer,
        "context_metadata": context_chunks
    })
    # Optionally save chat history to file
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state['chat_history'], f, ensure_ascii=False, indent=2)
    st.rerun() 