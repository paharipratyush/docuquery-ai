# DocuQuery AI

## Overview
DocuQuery AI is an advanced document question-answering system that leverages Retrieval-Augmented Generation (RAG) to provide intelligent responses from various document sources. Built with state-of-the-art NLP techniques and modern AI technologies, it offers a streamlined interface for extracting and querying information from multiple document formats.

## Features

### Document Processing
- **Multiple Format Support**: Process documents in various formats:
  - PDF files
  - Microsoft Word documents (DOCX)
  - Plain text files (TXT)
  - Web URLs
- **Smart Text Extraction**: Efficient text extraction with automatic content parsing
- **Chunk-based Processing**: Intelligent document chunking for optimal processing

### AI-Powered Analysis
- **Vector-based Search**: FAISS-powered vector store for efficient similarity search
- **Advanced Embeddings**: Utilizes HuggingFace's sentence transformers for text embeddings
- **Intelligent QA**: Leverages Meta's Llama model for generating accurate responses

### User Interface
- **Interactive Web Interface**: Built with Streamlit for a seamless user experience
- **Real-time Processing**: Dynamic document processing and question answering
- **Session Management**: Maintain context across multiple queries
- **Error Handling**: Robust error handling with informative feedback

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Hugging Face API key

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/docuquery-ai.git
cd docuquery-ai
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `secret_api_keys.py` in the project root:
```python
huggingface_api_key = "your-api-key-here"
```

## Project Structure
```
docuquery-ai/
├── src/
│   ├── config.py              # Configuration and constants
│   ├── app.py                 # Main Streamlit application
│   ├── processors/
│   │   └── document_processor.py  # Document processing logic
│   └── qa/
│       └── question_answerer.py   # QA chain implementation
├── requirements.txt
├── README.md
└── secret_api_keys.py         # API keys (not tracked in git)
```

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Select input type:
   - Upload a document (PDF/DOCX/TXT)
   - Paste a URL
   - Enter text directly

3. Process the input and wait for confirmation

4. Ask questions about your document in natural language

## Configuration

Key parameters can be adjusted in `src/config.py`:
- `MAX_FILE_SIZE`: Maximum allowed file size (default: 10MB)
- `CHUNK_SIZE`: Text chunk size for processing
- `CHUNK_OVERLAP`: Overlap between chunks
- `EMBEDDING_MODEL`: HuggingFace model for embeddings
- `LLM_MODEL`: Language model for question answering

## Dependencies
- `streamlit`: Web interface
- `langchain`: Document processing and QA chains
- `faiss-cpu`: Vector similarity search
- `PyPDF2`: PDF processing
- `python-docx`: DOCX processing
- `huggingface-hub`: AI model access
- `sentence-transformers`: Text embeddings

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- HuggingFace for providing the model infrastructure
- Streamlit for the web framework
- FAISS for vector similarity search
- LangChain for document processing capabilities

## Contact
For questions and support, please open an issue in the GitHub repository.
