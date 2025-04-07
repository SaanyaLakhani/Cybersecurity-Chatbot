# import os
# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # Set Gemini API Key
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"  # Replace with your actual Gemini API key

# # Streamlit Page Configuration
# st.set_page_config(page_title="Cybersecurity Q&A", page_icon="🔐", layout="wide")

# # Load custom CSS
# def load_css():
#     bg_image = "cyber1.jpg"  # Make sure this file is in the same directory
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: url("{bg_image}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#         }}
#         .stTitle {{
#             text-align: center;
#             font-size: 36px;
#             font-weight: bold;
#             color: #ffffff;
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
#         }}
#         .stTextInput>div>div>input {{
#             background-color: #ffffff;
#             color: #333;
#             font-size: 18px;
#             padding: 10px;
#             border-radius: 10px;
#             border: 1px solid #ccc;
#         }}
#         .stButton>button {{
#             background-color: #ff4b4b;
#             color: white;
#             font-size: 18px;
#             padding: 10px 20px;
#             border-radius: 8px;
#             border: none;
#             transition: all 0.3s ease-in-out;
#         }}
#         .stButton>button:hover {{
#             background-color: #cc0000;
#         }}
#         .stSubheader {{
#             font-size: 24px;
#             font-weight: bold;
#             color: #ffffff;
#             text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
#         }}
#         .stMarkdown {{
#             font-size: 18px;
#             background-color: rgba(255, 255, 255, 0.8);
#             padding: 15px;
#             border-radius: 10px;
#         }}
#         .footer {{
#             position: fixed;
#             bottom: 10px;
#             width: 100%;
#             text-align: center;
#             font-size: 14px;
#             color: white;
#             text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Apply the CSS
# load_css()

# # Title
# st.markdown('<h1 class="stTitle">🔐 Cybersecurity Q&A System</h1>', unsafe_allow_html=True)
# st.write("Ask cybersecurity-related questions and get answers using AI-powered retrieval.")

# # Initialize Gemini LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0")

# # Initialize Gemini embeddings
# embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"
#     )

# # Load FAISS vector store
# vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# # Create the RAG-based Q&A system
# qa_system = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(),
#     return_source_documents=True
# )

# # User input
# query = st.text_input("🔍 Enter your cybersecurity-related question:")

# if st.button("Get Answer"):
#     if query:
#         with st.spinner("Fetching answer..."):
#             response = qa_system.invoke({"query": query})

#             # Display answer
#             st.markdown('<h2 class="stSubheader">💡 Answer:</h2>', unsafe_allow_html=True)
#             st.markdown(f'<div class="stMarkdown">{response["result"]}</div>', unsafe_allow_html=True)

#             # Display sources correctly
#             st.markdown('<h2 class="stSubheader">📚 Sources:</h2>', unsafe_allow_html=True)
#             sources = set()  # Avoid duplicate sources
#             for doc in response["source_documents"]:
#                 source_name = doc.metadata.get("source", "Unknown")
#                 sources.add(source_name)

#             for src in sources:
#                 st.write(f"- {src}")
#     else:
#         st.warning("Please enter a question before clicking 'Get Answer'.")

# # Footer
# st.markdown('<div class="footer">🔒 <b>Cybersecurity AI Assistant</b> | Built with LangChain & Gemini</div>', unsafe_allow_html=True)

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

# Load custom CSS
def load_css():
    bg_image = "https://cdn.pixabay.com/photo/2020/04/10/15/59/cyber-security-5022845_1280.jpg"  # Cybersecurity-themed background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stTitle {{
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #00ffcc; /* Adjusted color for visibility */
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }}
        .stTextInput>div>div>input {{
            background-color: #ffffff;
            color: #000000;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ccc;
        }}
        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease-in-out;
        }}
        .stButton>button:hover {{
            background-color: #cc0000;
        }}
        .stSubheader {{
            font-size: 24px;
            font-weight: bold;
            color: #00ffcc; /* Adjusted color for visibility */
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }}
        .stMarkdown {{
            font-size: 18px;
            background-color: rgba(0, 0, 0, 0.8);
            color: #ffffff;
            padding: 15px;
            border-radius: 10px;
        }}
        .footer {{
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: #00ffcc;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the CSS
load_css()

# Title
st.markdown('<h1 class="stTitle">🔐 Cybersecurity Q&A System</h1>', unsafe_allow_html=True)
st.write("Ask cybersecurity-related questions and get answers using AI-powered retrieval.")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0")

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"
    )

# Load FAISS vector store
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

            # Display answer
            st.markdown('<h2 class="stSubheader">💡 Answer:</h2>', unsafe_allow_html=True)
            st.markdown(f'<div class="stMarkdown">{response["result"]}</div>', unsafe_allow_html=True)

            # Display sources correctly
            st.markdown('<h2 class="stSubheader">📚 Sources:</h2>', unsafe_allow_html=True)
            sources = set()  # Avoid duplicate sources
            for doc in response["source_documents"]:
                source_name = doc.metadata.get("source", "Unknown")
                sources.add(source_name)

            for src in sources:
                st.write(f"- {src}")
    else:
        st.warning("Please enter a question before clicking 'Get Answer'.")

# Footer
st.markdown('<div class="footer">🔒 <b>Cybersecurity AI Assistant</b> | Built with LangChain & Gemini</div>', unsafe_allow_html=True)
