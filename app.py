import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile

# --- পেজ কনফিগারেশন এবং টাইটেল ---
st.set_page_config(page_title="GovAI Pro (Llama 3)", page_icon="⚖️", layout="wide")

st.title("⚖️ GovAI Pro (Llama 3 Edition)")
st.markdown("""
এই অ্যাপ্লিকেশনটি Llama 3 এবং সর্বাধুনিক AI প্রযুক্তি ব্যবহার করে যেকোনো দেশের আইনি ডকুমেন্ট, সংবিধান, টেন্ডার বা সরকারি নীতিমালা বিশ্লেষণ করতে পারে। 
এটি বাংলা এবং ইংরেজি উভয় ভাষাতেই কাজ করতে সক্ষম।
""")

# --- সাইডবার: API কী, মডেল নির্বাচন এবং ফাইল আপলোড ---
with st.sidebar:
    st.header("🛠️ কনফিগারেশন")
    st.markdown("""
    এই অ্যাপটি চালানোর জন্য আপনার একটি Groq API Key প্রয়োজন হবে। 
    [এখান থেকে বিনামূল্যে আপনার Groq API Key নিন](https://console.groq.com/keys)।
    """)
    
    # Groq API Key ইনপুট
    groq_api_key = st.text_input("🔑 আপনার Groq API Key দিন", type="password", placeholder="gsk_...")
    
    st.markdown("---")
    
    # LLM মডেল নির্বাচন
    llm_model = st.selectbox(
        "🧠 LLM মডেল বেছে নিন",
        ("Llama3-70b-8192", "Llama3-8b-8192"),
        index=0,
        help="70B মডেল বেশি শক্তিশালী কিন্তু কিছুটা ধীর। 8B মডেল খুব দ্রুত কাজ করে।"
    )
    
    st.markdown("---")
    
    # PDF ফাইল আপলোড
    uploaded_file = st.file_uploader("📄 আপনার PDF ডকুমেন্টটি আপলোড করুন (বাংলা/ইংরেজি)", type="pdf")
    
    st.markdown("---")
    st.info("আপনার ডেটা সম্পূর্ণ সুরক্ষিত। API Key বা ডকুমেন্টের কোনো তথ্য আমরা সংরক্ষণ করি না।")

# --- ক্যাশিং ফাংশন (পারফরম্যান্স বাড়ানোর জন্য) ---

# বহুভাষিক এমবেডিং মডেলটি একবারই লোড হবে
@st.cache_resource
def load_multilingual_embeddings():
    st.info("বহুভাষিক এমবেডিং মডেল (bge-m3) লোড করা হচ্ছে... (প্রথমবার একটু সময় লাগতে পারে)")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# ভেক্টর স্টোর তৈরি এবং ক্যাশ করা
# CORRECTED FUNCTION
@st.cache_data(show_spinner="ডকুমেন্ট প্রসেস করা হচ্ছে...")
def create_vector_store(_file_content):
    if not _file_content:
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_file_content)
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # এমবেডিং মডেলটি এই ফাংশনের ভেতরে কল করা হচ্ছে
    embeddings = load_multilingual_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.remove(tmp_file_path)
    return vectorstore

# --- মূল অ্যাপ্লিকেশন লজিক ---

if not groq_api_key:
    st.warning("👈 অনুগ্রহ করে সাইডবারে আপনার Groq API Key দিন।")
elif not uploaded_file:
    st.warning("👈 অনুগ্রহ করে সাইডবারে একটি PDF ফাইল আপলোড করুন।")
else:
    # ফাইল কন্টেন্ট পড়া
    file_content = uploaded_file.getvalue()
    
    # ভেক্টর স্টোর তৈরি বা ক্যাশ থেকে লোড করা
    # CORRECTED FUNCTION CALL
    vectorstore = create_vector_store(file_content)
    
    if vectorstore:
        st.success(f"✅ ডকুমেন্ট সফলভাবে প্রসেস করা হয়েছে। এখন আপনি '{llm_model}' ব্যবহার করে প্রশ্ন করতে পারেন।")

        # ব্যবহারকারীর প্রশ্ন ইনপুট
        query = st.text_input(
            "❓ ডকুমেন্ট সম্পর্কে আপনার প্রশ্ন এখানে লিখুন (বাংলা বা ইংরেজিতে):",
            placeholder="What are the key liabilities for the contractor? / ஒப்பந்தকারীর মূল দায়বদ্ধতাগুলো কী কী?"
        )

        if st.button("تحليل করুন (Analyze)"):
            if not query:
                st.error("অনুগ্রহ করে একটি প্রশ্ন লিখুন।")
            else:
                with st.spinner(f"Llama 3 আপনার প্রশ্নের উত্তর খুঁজছে... অনুগ্রহ করে অপেক্ষা করুন।"):
                    try:
                        # LLM মডেল ইনিশিয়ালাইজ করা
                        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=llm_model)

                        # বহুভাষিক প্রম্পট টেমপ্লেট
                        prompt_template = """
                        You are "GovAI Pro", a highly intelligent legal and policy analysis AI.
                        Your task is to answer the user's question based ONLY on the provided context from a legal or government document.
                        The document and the query can be in either English or Bengali. You MUST answer in the same language as the user's query.

                        Your answer must be structured into three distinct sections:
                        1.  **Summary (সারাংশ):** Provide a clear and concise answer to the user's question in simple language.
                        2.  **Relevant Clauses (সম্পর্কিত ধারা):** Mention the specific clause or section numbers from the document that support your answer.
                        3.  **Potential Risks/Key Points (সম্ভাব্য ঝুঁকি/গুরুত্বপূর্ণ বিষয়):** Highlight any risks, penalties, deadlines, obligations, or inconsistencies mentioned in the context related to the query.

                        If the answer is not found in the provided context, you MUST state: "The provided context does not contain information on this topic. / প্রদত্ত কনটেক্সটে এই বিষয়ে কোনো তথ্য পাওয়া যায়নি।"

                        CONTEXT:
                        {context}

                        QUERY:
                        {question}

                        Follow these instructions and provide a structured, factual answer.
                        """
                        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                        # RetrievalQA চেইন তৈরি করা
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(),
                            chain_type_kwargs={"prompt": PROMPT}
                        )

                        # ফলাফল জেনারেট করা
                        result = qa_chain.invoke({"query": query})

                        # ফলাফল প্রদর্শন
                        st.subheader("📄 GovAI Pro বিশ্লেষণ:")
                        st.markdown(result["result"])

                    except Exception as e:
                        st.error(f"একটি ত্রুটি ঘটেছে: {e}")
                        st.info("আপনার Groq API Key সঠিক কিনা যাচাই করুন অথবা অন্য কোনো প্রশ্ন চেষ্টা করুন।")
