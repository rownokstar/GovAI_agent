import streamlit as st
import os
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fpdf import FPDF
from docx import Document as DocxDocument
import pandas as pd
import io

# --- পেজ কনফিগারেশন ---
st.set_page_config(page_title="GovAI Pro 2.0", page_icon="🤖", layout="wide")
st.title("🤖 GovAI Pro 2.0 - আপনার ইন্টারেক্টিভ ডকুমেন্ট অ্যাসিস্ট্যান্ট")

# --- সাইডবার ---
with st.sidebar:
    st.header("🛠️ কনফিগারেশন")
    st.markdown("API Key-গুলো এখন Streamlit Secrets থেকে স্বয়ংক্রিয়ভাবে লোড হচ্ছে।")
    
    if st.button("💬 নতুন চ্যাট শুরু করুন", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.info("এই অ্যাপটি বাংলা, ইংরেজি এবং বাংলিশ ভাষায় কাজ করতে পারে।")


# --- API কী লোড করা ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not groq_api_key or not openai_api_key:
    st.error("অনুগ্রহ করে Streamlit Cloud-এর Settings > Secrets সেকশনে আপনার Groq এবং OpenAI API Key যোগ করুন।")
    st.stop()

# --- ফাংশনসমূহ ---

@st.cache_resource(show_spinner="ডকুমেন্ট প্রসেস করা হচ্ছে... ⏳")
def get_vectorstore_from_file(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file_path = f"./{uploaded_file.name}"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".xls": UnstructuredExcelLoader,
            ".xlsx": UnstructuredExcelLoader
        }
        loader_class = loaders.get(file_extension)

        if loader_class is None:
            st.error("এই ফাইল ফরম্যাটটি সাপোর্ট করে না।")
            return None
            
        loader = loader_class(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_documents(document_chunks, embeddings)

        os.remove(temp_file_path)
        return vector_store
    except Exception as e:
        st.error(f"ফাইল প্রসেসিং-এ সমস্যা হয়েছে: {e}")
        return None

def get_history_aware_retriever(llm, retriever):
    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a follow-up question, rephrase the question to be a standalone question. **Crucially, the rephrased question MUST be in the same language as the original follow-up question (English, Bengali, or Banglish).** Do not translate the question."),
        ("user", "{input}"),
    ])
    return create_history_aware_retriever(llm, retriever, history_aware_prompt)

def get_conversational_rag_chain(history_aware_retriever):
    llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="Llama3-70b-8192")
    
    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and friendly AI assistant named GovAI Pro. Answer the user's questions based on the provided context and conversation history.

Language Rules:
- If the user asks in English, respond in English.
- If the user asks in Bengali (বাংলা), respond in clear Bengali.
- If the user asks in Banglish (e.g., "amar kisu document lagbe"), understand it as Bengali and respond in clear Bengali.

Formatting Rules:
- Use appropriate emojis to make the conversation engaging. 👍
- If you need to present structured data (like comparisons, lists of items, etc.), use Markdown tables.
- If you don't know the answer from the context, politely say so in the correct language.

Context:
{context}"""),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, conversational_prompt)
    return create_retrieval_chain(history_aware_retriever, stuff_documents_chain)

# --- রিপোর্ট জেনারেট ফাংশন ---
def generate_pdf_report(chat_history):
    pdf = FPDF()
    pdf.add_page()
    try:
        # This path works in Streamlit Cloud's default environment
        pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
    except RuntimeError:
        # Fallback for local environments that might not have the font
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, "Warning: DejaVuSans font not found. Using Arial (non-Unicode).", ln=True)

    pdf.cell(0, 10, txt="GovAI Pro - Chat Report", ln=True, align='C')
    for message in chat_history:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        # Encode to a basic encoding and decode back to handle potential problematic characters
        content = message.content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, f"{role}: {content}")
        pdf.ln(5)
    
    # CORRECTED LINE: pdf.output() now returns bytes directly.
    return pdf.output()

def generate_docx_report(chat_history):
    document = DocxDocument()
    document.add_heading('GovAI Pro - Chat Report', 0)
    for message in chat_history:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        document.add_paragraph(f"{role}: {message.content}")
    bio = io.BytesIO()
    document.save(bio)
    return bio.getvalue()

def generate_excel_report(chat_history):
    data = {"Role": [], "Message": []}
    for message in chat_history:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        data["Role"].append(role)
        data["Message"].append(message.content)
    df = pd.DataFrame(data)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Chat History')
    return bio.getvalue()

# --- সেশন স্টেট শুরু করা ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="नमस्ते! আমি GovAI Pro। আপনার ডকুমেন্ট আপলোড করুন এবং প্রশ্ন জিজ্ঞাসা শুরু করুন।")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- ফাইল আপলোড এবং প্রসেসিং ---
uploaded_file = st.file_uploader(
    "📄 আপনার ডকুমেন্ট আপলোড করুন (PDF, DOCX, CSV, Excel)", 
    type=['pdf', 'docx', 'csv', 'xls', 'xlsx'],
    help="এখানে ফাইল আপলোড করার পর আপনি প্রশ্ন করতে পারবেন।"
)

if uploaded_file:
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        st.session_state.vector_store = get_vectorstore_from_file(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name

if st.session_state.vector_store is None:
    st.info("একটি ডকুমেন্ট আপলোড করে শুরু করুন।")
else:
    # --- চ্যাট হিস্ট্রি ডিসপ্লে ---
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👤"):
                st.write(message.content)

    # --- ইউজার ইনপুট ---
    user_query = st.chat_input("আপনার প্রশ্ন এখানে লিখুন...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human", avatar="👤"):
            st.write(user_query)

        with st.chat_message("AI", avatar="🤖"):
            llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
            retriever = st.session_state.vector_store.as_retriever()
            history_aware_retriever = get_history_aware_retriever(llm, retriever)
            rag_chain = get_conversational_rag_chain(history_aware_retriever)
            
            def stream_chunks():
                response_stream = rag_chain.stream({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                for chunk in response_stream:
                    if "answer" in chunk:
                        yield chunk["answer"]
                        time.sleep(0.02)
            
            full_response = st.write_stream(stream_chunks)
            st.session_state.chat_history.append(AIMessage(content=full_response))

            # --- ডাউনলোড বাটন ---
            st.markdown("---")
            st.write("এই আলোচনাটি ডাউনলোড করুন:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📄 PDF", generate_pdf_report(st.session_state.chat_history), "chat_report.pdf", "application/pdf", use_container_width=True)
            with col2:
                st.download_button("📑 DOCX", generate_docx_report(st.session_state.chat_history), "chat_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
            with col3:
                st.download_button("📊 Excel", generate_excel_report(st.session_state.chat_history), "chat_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
