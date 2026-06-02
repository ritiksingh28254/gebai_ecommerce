import streamlit as st
from rag_pipeline import build_rag_from_folder, get_answer
from model_loader import load_model
import os

st.set_page_config(
    page_title="E-commerce Product Assistant",
    page_icon="🛍️",
    layout="centered"
)

st.title("E-commerce Product Assistant")
st.write("Ask me anything about our products, returns, warranty and more!")

# Load model once
@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_retriever():
    return build_rag_from_folder("D:\gebai_ecommerce\data")

llm = get_model()

with st.spinner("🔄 Loading product knowledge base..."):
    retriever = get_retriever()

st.success("✅ Assistant is ready!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
question = st.chat_input("Ask your question here...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question, retriever, llm)
        st.write(answer)

        # Show sources
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(sources):
                st.write(f"**Source {i+1}:** {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})