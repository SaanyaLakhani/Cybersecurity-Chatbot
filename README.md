
# 🔐 Cybersecurity Chatbot using LangChain + Gemini

This is an intelligent chatbot designed to answer technical cybersecurity-related questions using trusted sources like NIST and ENISA reports. It leverages **Google's Gemini-Pro model**, **LangChain**, and **FAISS** for Retrieval-Augmented Generation (RAG), with an easy-to-use interface built in **Streamlit**.

---

## 🚀 Features

- Ask cybersecurity questions based on real-world reports and standards.
- Uses Retrieval-Augmented Generation (RAG) for accurate, grounded answers.
- Embeds documents using Google's `models/embedding-001`.
- Responses generated using the `gemini-pro` chat model.
- Clean and interactive UI built with Streamlit.

---

## 📁 Project Structure

```
LLM Proj1/
│
├── app.py                  👉 Main Streamlit app to run the chatbot
├── rag.py                  👉 Builds and queries the vector database using FAISS
├── llm.py                  👉 Connects to Gemini-Pro using LangChain
├── test.py                 👉 (Optional) Used for testing single queries
├── docs/                   👉 Folder containing cybersecurity PDFs (NIST, ENISA, etc.)
├── text_files/             👉 Converted text files from the PDFs
├── faiss_index/            👉 Stored FAISS index for document search
├── requirements.txt        👉 All Python dependencies
└── README.md               👉 You're here!
```

---

## 🧠 Models Used

| Purpose          | Model Name               | Description                                |
|------------------|--------------------------|--------------------------------------------|
| Embedding Model  | `models/embedding-001`   | Converts docs & queries into vector form   |
| Chat Model (LLM) | `gemini-pro`             | Generates final response with LangChain    |

---

## 📚 Dataset

Documents used include:
- NIST Special Publications (SP 800-53, SP 800-37, etc.)
- ENISA Threat Landscape Reports

These are stored in the `/docs` folder and processed into chunks using LangChain.

---

## 🛠️ How It Works

1. **Document Ingestion:**
   - Cybersecurity PDFs are converted to text using LangChain loaders.
   - Text is split into manageable chunks.

2. **Embeddings:**
   - Each chunk is embedded using Google’s `models/embedding-001`.

3. **Vector Store:**
   - FAISS is used to store embeddings and enable similarity search.

4. **User Query:**
   - A user's question is embedded and matched to the most similar chunks.
   - These chunks are passed to `gemini-pro` via LangChain for answer generation.

---

## ▶️ Running the App

### 1. 🔧 Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. 🔑 Set API Key
In your terminal:
```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### 3. 🧠 Build Vector Index (Only once)
```bash
python rag.py
```

### 4. 🚀 Launch the Chatbot
```bash
streamlit run app.py
```

---

## 💬 Sample Questions to Try

- What are the five functions of the NIST Cybersecurity Framework?
- What does ENISA say about ransomware attacks?
- How does NIST define risk assessment and risk management?
- What steps should be taken for incident response according to NIST?

---

## 🧪 Tech Stack

- **LangChain** – Framework for LLM applications
- **Google Gemini Pro** – LLM used for generating answers
- **models/embedding-001** – Embedding model for semantic similarity
- **FAISS** – Vector database for fast similarity search
- **Streamlit** – Interactive UI
- **Python** – Backend programming

---

## 📌 Notes

- Make sure to convert new PDFs to `.txt` and run `rag.py` again if documents are updated.
- You can test individual questions using `test.py` before deploying to the app.

---

## ✨ Future Improvements

- Add chat history
- Include feedback mechanism
- Connect more cybersecurity datasets or real-time threat feeds

---

## 🧑‍💻 Author

Developed by [Your Name]

---

## 🛡️ License

This project is for educational/demo purposes. Review NIST/ENISA licensing for document reuse.
