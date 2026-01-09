import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "lokeshch19/ModernPubMedBERT"
LLM_MODEL = "llama3.2"

# 1. Initialize Components
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cuda'}
)

# Connect to the existing ChromaDB
vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# means get the top 3 most relevant chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize the Llama 3.2 model via Ollama
llm = ChatOllama(model=LLM_MODEL)

# 2. Define the RAG Prompt Template
template = """
You are an AI medical research assistant. 
Answer the user's question based ONLY on the following context.
If the context doesn't contain the answer, state that you cannot find the answer in the provided documents.
Do not use any prior knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Define the RAG Chain

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Create the Streamlit App
st.set_page_config(page_title="AI Medical Research Assistant", layout="wide")
st.title("AI Medical Research Assistant (RAG)")
st.write("Ask questions about your medical journals")

# We use session state to keep track of the last query and its results
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
    st.session_state.last_answer = ""
    st.session_state.last_sources = []

# Input box for the user's question
question = st.text_input("Enter your question:", key="query_input")

if st.button("Get Answer"):
    if question:
        st.session_state.last_query = question
        
        with st.spinner("Finding answers..."):

            st.session_state.last_answer = rag_chain.invoke(question)

            st.session_state.last_sources = retriever.invoke(question)

# Display the results
if st.session_state.last_query:
    st.subheader("Question:")
    st.write(st.session_state.last_query)
    
    st.subheader("Answer:")
    st.write(st.session_state.last_answer)
    
    st.subheader("Sources (Citations):")
    if st.session_state.last_sources:
        for i, doc in enumerate(st.session_state.last_sources):
            with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'N/A')}"):
                st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                st.write("**Content:**")
                st.write(doc.page_content)
    else:
        st.write("No sources found for this query.")