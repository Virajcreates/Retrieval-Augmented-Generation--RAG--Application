Retrieval-Augmented Generation (RAG) System
Overview
This project implements a Retrieval-Augmented Generation (RAG) system for processing and querying documents (PDF, TXT, DOCX) and web content using natural language processing. Built with FastAPI and Supabase, it supports secure document uploads, URL scraping, and context-based query responses. It integrates Hugging Face’s Zephyr-7B model with 8-bit quantization for CPU-only execution (~6-7GB RAM, ~1-2 tokens/s), using LangChain and FAISS for efficient document chunking, embedding, and retrieval.
Features

Document Processing: Upload and process PDF, TXT, and DOCX files.
Web Scraping: Extract text from URLs using BeautifulSoup.
Query Answering: Context-based responses using Zephyr-7B and LangChain.
Authentication: Secure user authentication via Supabase with JWT.
Scalable API: FastAPI backend with CORS for frontend integration.
CPU Optimization: 8-bit quantization for efficient CPU execution.

Technologies

Python, FastAPI, Supabase
LangChain, Hugging Face Transformers, FAISS, PyTorch
BeautifulSoup, Requests

Setup Instructions

Clone the Repository:
git clone <https://github.com/Virajcreates/Retrieval-Augmented-Generation--RAG--Application>
cd <repository-directory>


Install Dependencies:Ensure Python 3.8+ is installed, then:
pip install -r requirements.txt

Or install manually:
pip install fastapi uvicorn supabase langchain langchain-huggingface transformers==4.45.2 pyzmq python-docx requests beautifulsoup4 validators tqdm bitsandbytes==0.43.3 accelerate python-dotenv torch --index-url https://download.pytorch.org/whl/cpu

For Windows, if bitsandbytes fails:
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui


Set Environment Variables:Create a .env file:
SUPABASE_URL=https://your-supabase-url.supabase.co
SUPABASE_KEY=your-supabase-anon-key


Run the Application:
python backend.py

The API will run on http://localhost:8000.


Usage

Upload a Document:
curl -X POST -H "Authorization: Bearer <your-jwt-token>" -F "file=@document.pdf" http://localhost:8000/upload


Add a URL:
curl -X POST -H "Authorization: Bearer <your-jwt-token>" -F "url=https://example.com" http://localhost:8000/add_url


Query Documents:
curl -X POST -H "Authorization: Bearer <your-jwt-token>" -H "Content-Type: application/json" -d '{"question": "What is the mission of IEEE?"}' http://localhost:8000/query


View Documents:
curl -X GET -H "Authorization: Bearer <your-jwt-token>" http://localhost:8000/documents


Clean Up:
curl -X DELETE -H "Authorization: Bearer <your-jwt-token>" http://localhost:8000/cleanup



System Requirements

RAM: Minimum 8GB free (16GB recommended) for Zephyr-7B with 8-bit quantization (6-7GB), embeddings (1GB), and FastAPI/Supabase (~1GB).
OS: Windows/Linux/MacOS.
CPU: Multi-core CPU for efficient inference (~1-2 tokens/s).

Troubleshooting

Memory Issues: If RAM is insufficient, reduce max_new_tokens to 256 in backend.py or switch to distilgpt2 model (~1GB RAM).
Supabase Errors: Verify documents and interactions table permissions in Supabase for INSERT, SELECT, and DELETE with user_id matching auth.uid().
Logs: Enable verbose logging:export TRANSFORMERS_VERBOSITY=info
python backend.py



License
MIT License
