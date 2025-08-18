📄 Bank Statement Extraction Pipeline
Overview

This project implements an end-to-end bank statement parser with:

Streamlit Dashboard → Upload PDF, view account details & transactions, download CSV

FastAPI Backend → REST API endpoints (/extract, /health) for automated extraction

LLM (Gemini) Fallback → Ensures tables & details are extracted even when algorithmic parsing fails

Agentic-Doc + Heuristics → High accuracy on text-based and scanned PDFs

Features

🔒 PII-safe: Redact personal info before upload (PdfGear / Sejda recommended)

📑 Account Holder Details: Name, Address, Contact Number, Email

🏦 Bank Account Details: Account Number, IFSC, Branch Address

💰 Transactions Table: Date, Description, Debit, Credit, Balance

✅ Validation: IFSC regex, account number length, balance checks

📊 Dashboard: Searchable transactions, Debit/Credit totals, CSV download

🌐 API: /extract (upload PDF) and /health (status)

Tech Stack

Python 3.10+

Streamlit → Dashboard

FastAPI → API layer

Agentic-Doc → PDF parsing

Google Gemini 2.5 (Vertex AI) → LLM fallback for structured extraction

Pandas → Table handling

Uvicorn → API server

Project Structure
bank_statement/
│── app.py                 # Streamlit Dashboard
│── server.py              # FastAPI server
│── pipeline.py            # Core pipeline (PDF → structured JSON)
│── requirements.txt       # Dependencies
│── Procfile               # For Heroku deployment
│── render.yaml            # For Render deployment
│── README.md              # Project documentation
│── postman/
│   └── BankStatement.postman_collection.json  # API tests
│── .streamlit/
│   └── secrets.toml (for deployment, not committed)

Installation (Local)

Clone repo:

git clone https://github.com/Gowtham8309/bank_statement.git
cd bank_statement


Create virtual environment:

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

Environment Variables

For Gemini / Vertex AI integration, set these environment variables (locally or in cloud deployment):

GOOGLE_CLOUD_PROJECT=verdant-copilot-467518-c1
VERTEX_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=verdant-copilot-467518-c1-251370377dd2.json
USE_GEMINI=1
GOOGLE_API_KEY=your_google_api_key_here

🔹 Local setup

Download your Vertex AI service account key JSON.

Place it in your project directory.

Set path in GOOGLE_APPLICATION_CREDENTIALS.

Example (PowerShell):

$env:GOOGLE_CLOUD_PROJECT="verdant-copilot-467518-c1"
$env:VERTEX_LOCATION="us-central1"
$env:GOOGLE_APPLICATION_CREDENTIALS='/abs/path/verdant-copilot-467518-c1-251370377dd2.json'
$env:USE_GEMINI="1"
$env:GOOGLE_API_KEY="AIzaSyXXXXXXX"

Running Locally
🔹 Streamlit Dashboard
streamlit run app.py


Access at → http://localhost:8501

🔹 FastAPI Server
uvicorn server:app --reload --port 8000


Endpoints:

Health: http://localhost:8000/health

Extract:

curl -F "file=@unlocked.pdf" "http://localhost:8000/extract?llm_fallback=true&use_gemini=true"
