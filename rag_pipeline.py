from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def build_rag_from_folder(folder_path):
    all_documents = []

    # Load all PDFs from folder
    print("Loading all product documents...")
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            all_documents.extend(loader.load())
            print(f"Loaded: {file}")

    print(f"Total pages loaded: {len(all_documents)}")

    # Split into chunks
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Total chunks: {len(chunks)}")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

    vectorstore.persist()
    # Store in ChromaDB
    # print("Storing in ChromaDB...")
    # vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("RAG pipeline ready!")

    

    return retriever


def get_answer(question, retriever, llm):
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = " ".join([d.page_content for d in docs])

    # Build prompt
    prompt = f"""You are a helpful ecommerce customer support assistant.
Use the following product information to answer the customer's question accurately.

Product Information: {context}

Customer Question: {question}

Answer:"""

    answer = llm(prompt)[0]['generated_text']
    return answer, docs