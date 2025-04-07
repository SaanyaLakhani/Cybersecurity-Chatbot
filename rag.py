import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"  # Replace with your actual Gemini API key

# Streamlit Page Configuration
st.set_page_config(page_title="Cybersecurity Q&A", page_icon="🔐", layout="wide")

# Title
st.title("🔐 Cybersecurity Q&A System")
st.write("Ask cybersecurity-related questions and get answers using AI-powered retrieval.")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0")

# Load FAISS vector store
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"
    )
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create the RAG-based Q&A system
qa_system = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# User input
query = st.text_input("🔍 Enter your cybersecurity-related question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Fetching answer..."):
            response = qa_system.invoke({"query": query})
            
            st.subheader("💡 Answer:")
            st.write(response["result"])

            st.subheader("📚 Sources:")
            sources = set()
            for doc in response["source_documents"]:
                sources.add(doc.metadata.get("source", "Unknown"))

            for src in sources:
                st.write(f"- {src}")
    else:
        st.warning("Please enter a question before clicking 'Get Answer'.")

# Footer
st.markdown("---")
st.markdown("🔒 **Cybersecurity AI Assistant** | Built with LangChain & Gemini")
