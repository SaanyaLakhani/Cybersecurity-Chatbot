import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"  # Replace with actual Gemini API key

# Define input and output folders
input_folder = "resources"  # Folder containing PDFs
vector_store_path = "faiss_index"  # Path to store FAISS index

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyCu0aMmBo-wGew1POid7sN3RrXkJOibG_0"
    )

# List to store document chunks
all_chunks = []

# Process each PDF in the input folder
for pdf_file in os.listdir(input_folder):
    if pdf_file.endswith(".pdf"):
        print(f"Processing: {pdf_file}")

        # Load the PDF
        pdf_path = os.path.join(input_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Assign metadata (PDF name) to each document
        for doc in documents:
            doc.metadata["source"] = pdf_file  # Store PDF name as metadata

        # Split text into chunks
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)  # Store all processed chunks

print(f"Total Chunks Created: {len(all_chunks)}")

# Store chunks in FAISS (this will retain metadata)
vector_store = FAISS.from_documents(all_chunks, embeddings)

# Save FAISS index locally
vector_store.save_local(vector_store_path)

print("✅ All PDFs processed and FAISS index created successfully!")
