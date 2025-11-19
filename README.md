<div align="center">

<img src="https://readme-typing-svg.demolab.com?size=26&duration=3500&pause=1000&color=0F62FE&center=true&vCenter=true&width=700&lines=PGPR+Multimodal+RAG+%F0%9F%A7%AC+Text+%2B+Image;ML+Enrichment+%26+Streamlit+Dashboard;Fast+RAG+with+FAISS+%26+Ollama+%F0%9F%94%A5" alt="Typing Animation">

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
<a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.29%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
<a href="https://python.langchain.com/"><img src="https://img.shields.io/badge/LangChain-0.1%2B-1f6feb" alt="LangChain"></a>
<a href="#"><img src="https://img.shields.io/badge/Ollama-llama3.1-00A67E" alt="Ollama"></a>

</div>

PGPR Multimodal RAG System (with ML Enrichment)

Overview
- Retrieval-Augmented Generation (RAG) over PGPR scientific PDFs
- Multimodal: text + image analysis (ResNet50 features)
- ML enrichment: multiple classical models and a neural network to classify bacterial types
- Streamlit web UI: chat, image analysis, dataset statistics, and model comparison



Installation Walkthrough



On Windows (PowerShell)

1) Clone and enter the project


git clone <YOUR_REPO_URL>.git
cd pgpr-rag-local


2) Create and activate the virtual environment


python -m venv venv
venv\Scripts\activate


3) Install dependencies


pip install -r requirements.txt


4) Install and prepare Ollama (ensure llama3.1 is available)


ollama list
ollama pull llama3.1


5) First build (creates caches and models)


python build_enhanced_system.py


6) Launch the web app


streamlit run web_chatbot_enhanced.py

Then open http://localhost:8501


On macOS/Linux (bash/zsh)

1) Clone and enter the project


git clone <YOUR_REPO_URL>.git
cd pgpr-rag-local


2) Create and activate the virtual environment


python -m venv venv
source venv/bin/activate


3) Install dependencies


pip install -r requirements.txt


4) Install and prepare Ollama


curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1


5) Build and run


python build_enhanced_system.py
streamlit run web_chatbot_enhanced.py


Key Features
- Text RAG over PDFs stored in pgpr_docs/
- Image processing with feature extraction and similar-image search
- ML models: RandomForest, GradientBoosting, SVM, MLP, and a custom PyTorch network
- Caching: vector store and processed chunks saved under rag_cache/ (ignored from git)
- Web dashboard: chat, image analysis with confidences, metrics, and comparisons

Quickstart
1) Prerequisites
- Python 3.8+
- Ollama installed and llama3.1 model available
- Optional GPU for faster torch operations

2) Setup (Windows example)
1. Create and activate venv
   python -m venv venv
   venv\Scripts\activate
2. Install dependencies
   pip install -r requirements.txt
3. Ensure Ollama is running and llama3.1 is present
   ollama list
   ollama pull llama3.1

3) Build the system (first time)
- End-to-end builder (recommended)
  python build_enhanced_system.py

This will:
- Process PDFs from ./pgpr_docs/
- Extract image features from ./pgpr_images/images/
- Train/prepare ML models and cache metrics
- Create FAISS vector store
- Save caches to ./rag_cache/ and models to ./ml_models/

4) Run the web app
streamlit run web_chatbot_enhanced.py
Then open http://localhost:8501

Project Structure
- enhanced_multimodal_rag.py: Core multimodal RAG + ML enrichment
- web_chatbot_enhanced.py: Streamlit interface
- image_processor.py: Image analysis and feature extraction
- ml_model_builder.py: Training/loading ML models and metrics
- dataset_processor.py: CSV-based image dataset utilities
- build_enhanced_system.py: Orchestrates the initial build and caching
- retrain_ml_only.py, retrain_with_validation.py, quick_retrain.py: Model retraining utilities
- rag_cache/: Caches for vector store and chunks (ignored)
- ml_models/: Saved models and configs (ignored)
- pgpr_docs/: Input PDFs
- pgpr_images/images/: Image dataset (ignored by default)




Acknowledgements
- Built with LangChain, FAISS, PyTorch, scikit-learn, and Streamlit. Ollama provides local LLMs (llama3.1).


