PGPR Multimodal RAG System (with ML Enrichment)

Overview
- Retrieval-Augmented Generation (RAG) over PGPR scientific PDFs
- Multimodal: text + image analysis (ResNet50 features)
- ML enrichment: multiple classical models and a neural network to classify bacterial types
- Streamlit web UI: chat, image analysis, dataset statistics, and model comparison

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

2) Setup
On Windows PowerShell:
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

Data & Models
- Do not commit large datasets, cache files, or model weights. The .gitignore excludes:
  - venv/, rag_cache/, ml_models/, pgpr_images/images/
  - common model artifacts (*.pkl, *.pth, *.pt, *.faiss, etc.)
- Keep label CSV files (e.g., train_labels.csv, test_labels.csv) to reproduce experiments.

Environment & Secrets
- Store credentials and private settings in a local .env file (not committed).
- Do not check in API keys, private certs, or SSH keys. The .gitignore already excludes common secret patterns.

Troubleshooting
- Sklearn serialization mismatch: If you see CyHalfBinomialLoss errors, use the retrain scripts:
  python quick_retrain.py
  python retrain_ml_only.py
- If caches are missing: run the builder again:
  python build_enhanced_system.py

Licensing
- Add your preferred license (e.g., MIT) as LICENSE in the repo root.

Acknowledgements
- Built with LangChain, FAISS, PyTorch, scikit-learn, and Streamlit. Ollama provides local LLMs (llama3.1).


