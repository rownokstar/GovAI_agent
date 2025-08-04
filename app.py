import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile

# --- পেজ কনফিগারেশন এবং টাইটেল ---
st.set_page_config(page_title="GovAI Pro (Llama 3)", page_icon="⚖️", layout="wide")

st.title("⚖️ GovAI Pro (API Edition)")
st.markdown("""
এই অ্যাপ্লিকেশনটি Llama 3 এবং OpenAI-এর শক্তিশালী এমবেডিং ব্যবহার করে যেকোনো দেশের আইনি ডকুমেন্ট বিশ্লেষণ করে।
এটি এখন সম্পূর্ণ API-নির্ভর, তাই মেমোরি ক্র্যাশের কোনো সম্ভাবনা নেই।
""")

# --- সাইডবার: API কী, মডেল নির্বাচন এবং ফাইল আপলোড ---
with st.sidebar:
    st.header("🛠️ কনফিগারেশন")
    st.markdown("এই অ্যাপটি চালানোর জন্য আপনার দুটি API Key প্রয়োজন হবে।")
    
    # Groq API Key
    groq_api_key = st.text_input("🔑 Groq API Key (Llama 3-এর জন্য)", type="password", placeholder="gsk_...")
    
    # OpenAI API Key
    openai_api_key = st.text_input("🔑 OpenAI API Key (Embedding-এর জন্য)", type="password", placeholder="sk_...")

    st.markdown("---")
    
    # LLM মডেল নির্বাচন
    llm_model = st.selectbox(
        "🧠 LLM মডেল বেছে নিন",
        ("Llama3-70b-8192", "Llama3-8b-8192"),
        index=0
    )
    
    st.markdown("---")
    
    # PDF ফাইল আপলোড
    uploaded_file = st.file_uploader("📄 আপনার PDF ডকুমেন্টটি আপলোড করুন", type="pdf")
    
    st.markdown("---")
    st.info("আপনার ডেটা সম্পূর্ণ সুরক্ষিত।")

# --- ক্যাশিং ফাংশন ---

# ভেক্টর স্টোর তৈরি এবং ক্যাশ করা
@st.cache_data(show_spinner="ডকুমেন্ট প্রসেস করা হচ্ছে...")
def create_vector_store(_file_content, _openai_api_key):
    if not _file_content:
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_file_content)
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # OpenAI Embedding API ব্যবহার করা হচ্ছে
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=_openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"এমবেডিং তৈরির সময় ত্রুটি: {e}")
        return None

    os.remove(tmp_file_path)
    return vectorstore

# --- মূল অ্যাপ্লিকেশন লজিক ---

if not groq_api_key or not openai_api_key:
    st.warning("👈 অনুগ্রহ করে সাইডবারে আপনার Groq এবং OpenAI API Key দুটিই দিন।")
elif not uploaded_file:
    st.warning("👈 অনুগ্রহ করে সাইডবারে একটি PDF ফাইল আপলোড করুন।")
else:
    file_content = uploaded_file.getvalue()
    vectorstore = create_vector_store(file_content, openai_api_key)
    
    if vectorstore:
        st.success(f"✅ ডকুমেন্ট সফলভাবে প্রসেস করা হয়েছে। এখন আপনি '{llm_model}' ব্যবহার করে প্রশ্ন করতে পারেন।")
        query = st.text_input(
            "❓ ডকুমেন্ট সম্পর্কে আপনার প্রশ্ন লিখুন (বাংলা বা ইংরেজিতে):",
            placeholder="What are the key liabilities? / মূল দায়বদ্ধতাগুলো কী কী?"
        )

        if st.button("تحليل করুন (Analyze)"):
            if not query:
                st.error("অনুগ্রহ করে একটি প্রশ্ন লিখুন।")
            else:
                with st.spinner(f"Llama 3 আপনার প্রশ্নের উত্তর খুঁজছে..."):
                    try:
                        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=llm_model)
                        
                        retriever = vectorstore.as_retriever()
                        
                        prompt_template = """
                        You are "GovAI Pro". Your task is to answer the user's question based ONLY on the provided context.
                        The document and query can be in English or Bengali. You MUST answer in the same language as the user's query.

                        Structure your answer in three sections:
                        1.  **Summary (সারাংশ):** Clear answer to the query.
                        2.  **Relevant Clauses (সম্পর্কিত ধারা):** Mention the clause numbers.
                        3.  **Potential Risks/Key Points (সম্ভাব্য ঝুঁকি/গুরুত্বপূর্ণ বিষয়):** Highlight risks, penalties, etc.

                        If the answer is not in the context, state: "The provided context does not contain information on this topic. / প্রদত্ত কনটেক্সটে এই বিষয়ে কোনো তথ্য পাওয়া যায়নি।"

                        CONTEXT: {context}
                        QUERY: {question}
                        
                        Answer:
                        """
                        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            chain_type_kwargs={"prompt": PROMPT}
                        )

                        result = qa_chain.invoke({"query": query})
                        st.subheader("📄 GovAI Pro বিশ্লেষণ:")
                        st.markdown(result["result"])

                    except Exception as e:
                        st.error(f"একটি ত্রুটি ঘটেছে: {e}")
