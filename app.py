import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings  # Use HuggingFace for embeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("📚 Document Q&A Bot (Groq-powered)")

# Create an expander to group everything inside a collapsible box
with st.expander("Document Q&A Bot", expanded=True):
    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])

    # Initialize session state for vector store
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Process uploaded file
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load and process the PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Use a compatible embedding model for Qdrant
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = Qdrant.from_documents(
            documents=texts,
            embedding=embeddings,
            location=":memory:"  # In-memory storage for demo
        )
        
        st.success("Document processed successfully!")
        
        # Clean up
        os.remove("temp.pdf")

    # Question input
    if st.session_state.vector_store is not None:
        question = st.text_input("Ask a question about your document:")
        
        if question:
            # Use ChatGroq for answering questions
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever()
            )
            
            # Get answer
            with st.spinner("Generating answer..."):
                response = qa_chain.run(question)
                st.write("Answer:", response)

    # Instructions
    if st.session_state.vector_store is None:
        st.info("Please upload a PDF document to get started!")
