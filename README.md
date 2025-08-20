
# 🤖 GovAI Agent – Local LLM Government Assistant

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)](https://opensource.org)

**GovAI Agent** is an intelligent, privacy-focused AI assistant designed to answer questions about government schemes, acts, and citizen services. It leverages the power of local Large Language Models (LLMs) like Llama 3, combined with Retrieval-Augmented Generation (RAG) using FAISS, to provide accurate and contextual answers directly from your uploaded government-related datasets. No external API keys are required – everything runs locally on your machine.

> 🚀 **Upload Government Data** → 🤖 **AI Parses & Understands** → 💬 **Get Instant, Contextual Answers**



## 🌟 **Why GovAI Agent?**

<div align="center">

| Feature | Description |
|---------|-------------|
| 🛡️ **Privacy First** | All data processing and AI inference happen locally. Your sensitive government data never leaves your computer. |
| 🧠 **Smart Local AI** | Utilizes powerful open-source LLMs like Llama 3 for deep understanding without relying on external services. |
| 🔍 **RAG-Powered Accuracy** | Employs Retrieval-Augmented Generation to find relevant information in your documents before generating answers. |
| 💬 **Natural Interaction** | Ask questions in plain English about complex government documents and get clear, concise responses. |
| ⚡ **Dynamic Processing** | Automatically indexes and processes your uploaded CSV or text datasets for immediate querying. |

</div>



## 🎯 **Key Features**

### 📂 **Intelligent Dataset Handling**
- **Universal Upload**: Supports CSV and (planned) text document uploads.
- **Auto-Indexing**: Automatically processes and indexes data using Sentence Transformers and FAISS for fast retrieval.
- **Real-time RAG**: Context from your data is injected into the LLM for accurate answers.

### 💬 **Advanced AI Assistant**
- **Context-Aware Queries**: "What are the eligibility criteria for PMAY listed in this document?"
- **Summarization**: "Summarize the key points of the National Education Policy 2020."
- **Data Extraction**: "List all schemes related to agriculture and their funding agencies."
- **Smart Reasoning**: AI shows its step-by-step thinking process.

### 🛠️ **Technical Excellence**
- **RAG Architecture**: Combines retrieval (FAISS) with generation (Llama 3) for factual responses.
- **Local LLM Execution**: Runs Llama 3 (or similar) directly on your hardware using Hugging Face `transformers`.
- **Scalable Design**: Efficiently handles datasets of significant size.
- **Error Resilience**: Built-in handling for data and model loading issues.



## 🎯 **How It Works**


```mermaid
graph TD
    A[User Uploads Dataset] --> B[Auto Data Processing]
    B --> C[Text Embedding (Sentence Transformers)]
    C --> D[FAISS Vector Index Creation]
    D --> E[RAG Query Loop]
    E --> F[User Question]
    F --> G[Query Embedding]
    G --> H[FAISS Similarity Search]
    H --> I[Retrieve Relevant Context]
    I --> J[Llama 3 Prompt Engineering]
    J --> K[Local LLM Inference (Hugging Face)]
    K --> L[Generate Contextual Answer]
    L --> M[Display Answer + Thinking Steps]
## 🛠️ **Technology Stack**

<div align="center">

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | 🌐 Streamlit | Interactive web UI |
| **Backend** | 🐍 Python | Core logic & orchestration |
| **AI Engine** | 🤖 Hugging Face Transformers | Load & run Llama 3 |
| **Embeddings** | 🧠 Sentence Transformers | Create vector representations |
| **Search** | 🔍 FAISS | Fast similarity search on local data |
| **Data Processing** | 📦 Pandas/NumPy | Handle & manipulate datasets |

</div>

---

## 🚀 **Quick Start**

### 🔧 **Local Installation & Running**

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/rownokstar/GovAI_agent.git
    cd GovAI_agent
    ```

2.  **Set Up Python Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Crucial) Obtain Hugging Face Access & Token**
    *   Create a Hugging Face account: [https://huggingface.co/](https://huggingface.co/)
    *   Request access to the Llama 3 model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) on the Hugging Face website.
    *   Generate an Access Token: Go to your profile -> Settings -> Access Tokens -> New token.
    *   Log in via CLI:
        ```bash
        huggingface-cli login
        ```
        Paste your token when prompted. This allows the app to download the Llama 3 model.

5.  **Run the Application**
    ```bash
    streamlit run app.py
    ```
    Your default web browser should open automatically. If not, navigate to the URL provided in the terminal (usually `http://localhost:8501`).

---

## 📁 **Project Structure**

```
GovAI_agent/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore            # Specifies files/dirs to ignore in Git
```

---

## 🎯 **Smart Features Breakdown**

### 🧠 **Local LLM Inference**
- Runs powerful models like Llama 3 directly on your machine.
- Uses Hugging Face `transformers` and `pipeline` for easy loading and generation.
- Includes optional 4-bit quantization (`bitsandbytes`) to reduce memory requirements.

### 🔍 **RAG Pipeline**
- **Data Ingestion**: Uploaded CSV data is converted to text snippets.
- **Embedding**: Snippets are embedded using `all-MiniLM-L6-v2`.
- **Indexing**: Embeddings are stored in a FAISS index for fast search.
- **Retrieval**: User queries are embedded and matched against the index.
- **Generation**: Retrieved context is formatted and sent to the Llama 3 model.

### 💬 **Interactive & Transparent AI**
- **Thinking Process**: The app shows the AI's reasoning steps.
- **Natural Language**: Get answers in clear, understandable language.
- **Streaming Output**: Responses appear gradually for a better user experience.

---

## 🧪 **Try These Examples (After Uploading Data)**

- "What are the key features of the Ayushman Bharat scheme?"
- "Find all grants available for startups in the uploaded document."
- "Summarize the section on digital infrastructure."
- "List the ministries mentioned and their respective schemes."
- "What are the application deadlines for rural development funds?"

---

## 🎨 **User Interface Highlights**

### 📁 **Smart Sidebar**
- **Dataset Upload**: Simple drag-and-drop for CSV files.
- **Processing Status**: Real-time feedback on data loading and indexing.
- **Dataset Preview**: Quick glance at the structure of your uploaded data.

### 💬 **Chat Interface**
- **Natural Conversation**: Type your questions like you would ask a human expert.
- **Rich Responses**: Answers combined with the AI's reasoning steps.
- **History Tracking**: See your previous questions and the AI's responses.

---

## 🚀 **Advanced Capabilities**

### 🔧 **Configurable LLM**
- Easily switch between different Hugging Face models (e.g., Llama 3 variants, Mistral) by modifying the `model_name` in `app.py`.

### ⚡ **Performance Optimizations**
- **Quantization**: 4-bit quantization significantly lowers VRAM usage for Llama 3.
- **Caching**: `@st.cache_resource` ensures the LLM is loaded only once per session.
- **Device Mapping**: Automatically utilizes GPU (if available) for faster inference.

---

## 📸 **Screenshots** (Conceptual)

<div align="center">

### 📤 **Upload & Process Government Data**
![Dataset Upload](https://placehold.co/600x300/4CAF50/white?text=Upload+Gov+Dataset)

### 💬 **Ask Questions, Get AI-Powered Answers**
![Chat Interface](https://placehold.co/600x300/2196F3/white?text=AI+Chat+Interface)

### 🧠 **See the AI's Reasoning Process**
![Thinking Steps](https://placehold.co/600x300/FF9800/white?text=AI+Thinking+Process)

</div>

---

## 📋 **Requirements**

Ensure you have these installed:
- **Python 3.8+**
- **Git** (for cloning)
- **pip** (Python package installer)
- **(Recommended) CUDA-compatible GPU** for faster Llama 3 inference (CPU will work but be slow).

Key libraries installed via `requirements.txt`:
```txt
streamlit>=1.29.0
pandas>=1.5.0
numpy>=1.24.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
accelerate # For running large models
bitsandbytes # For 4-bit quantization (optional but recommended)
```

---

## 🤝 **Contributing**

We welcome contributions! 🎉

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

### 🎯 **Areas for Contribution**
- 📄 Support for PDF, TXT document uploads and parsing.
- 🌐 Multi-language support for queries and documents.
- 🎨 UI/UX enhancements and custom themes.
- 📚 Documentation improvements and examples.
- ⚙️ Adding more configuration options for the LLM and RAG pipeline.

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 DM Shahriar Hossain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🌟 **Show Your Support**

If you find this project useful:

- ⭐ **Star this repository**
- 🔄 **Share with colleagues**
- 🐛 **Report issues**
- 💡 **Suggest features**
- 🤝 **Contribute code**

---

## 🙌 **Credits & Acknowledgements**

### 👨‍💻 **Development**
- **Lead Developer**: [DM Shahriar Hossain](https://github.com/rownokstar)
- **AI Specialist**: Hugging Face Transformers Community
- **UI Framework**: Streamlit

### 🛠️ **Built With**
- **Python**: Core programming language.
- **Streamlit**: For rapid web application development.
- **FAISS**: For efficient vector similarity search.
- **Sentence Transformers**: For creating local embeddings.
- **Hugging Face Transformers**: For loading and running Llama 3.

### 🎨 **Inspiration**
- Making government information more accessible through AI.
- Privacy-preserving local data processing.
- Leveraging open-source AI for public good.

---

## 🚀 **Ready to Empower Citizens?**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rownokstar/GovAI_agent)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### 💡 **Unlock the Knowledge in Government Documents Today!**

</div>

---

<details>
<summary>🔍 <b>Technical Details</b></summary>

### 🧠 **AI Architecture**
- **Default LLM**: `meta-llama/Meta-Llama-3-8B-Instruct` (configurable)
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Search Engine**: FAISS FlatL2 Index
- **Processing Pipeline**: Dynamic RAG with local LLM

### ⚡ **Performance Considerations**
- **Model Loading**: First run will download the Llama 3 model (2-5 mins depending on internet).
- **Inference Speed**: Depends heavily on hardware. GPU (especially with quantization) is *much* faster than CPU.
- **Memory Usage**: Llama 3 8B quantized (~6-8 GB RAM/VRAM), Full precision (~16+ GB).
- **Scalability**: FAISS handles large local datasets efficiently.

### 🛡️ **Security & Privacy**
- **Local Execution**: No data uploaded to external servers.
- **Token Storage**: Hugging Face token stored locally via `huggingface-cli login`.
- **Data Handling**: Uploaded files are processed in memory and not persisted.

</details>

---

<details>
<summary>📦 <b>Deployment Options</b></summary>

### ☁️ **Cloud Deployment**
- **Challenges**: Running Llama 3 in the cloud requires significant resources (CPU/RAM or GPU).
- **Platforms**: Possible on platforms like RunPod, Vast.ai, or dedicated servers, but not typical free tiers.
- **Consideration**: API-based LLMs (like OpenAI) are easier for cloud deployment but compromise privacy.

### 🖥️ **Local Deployment**
- **Windows/Mac/Linux**: Fully supported.
- **Docker**: Can be containerized (requires careful handling of model caching).
- **Virtual Environment**: Strongly recommended for dependency isolation.
- **Production Ready**: Suitable for local, secure, expert use.

</details>

---

<div align="center">

### 🤖 **GovAI Agent - Bridging Citizens and Information**

[![Star](https://img.shields.io/github/stars/rownokstar/GovAI_agent?style=social)](https://github.com/rownokstar/GovAI_agent)
[![Fork](https://img.shields.io/github/forks/rownokstar/GovAI_agent?style=social)](https://github.com/rownokstar/GovAI_agent)
[![Issues](https://img.shields.io/github/issues/rownokstar/GovAI_agent?style=social)](https://github.com/rownokstar/GovAI_agent/issues)

</div>
```
