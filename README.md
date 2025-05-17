AI-Powered Question-Answer Generation with Mistral 7B Instruct
🔍 Overview
This project leverages Mistral 7B Instruct from Hugging Face to automatically generate question-answer (QA) pairs from documents. It features:

A FastAPI backend for inference and document handling.

A Streamlit frontend for easy user interaction.

Organized folder structure for HTML templates, uploaded documents, and generated output.

🗂️ Project Structure
php
Copy
Edit
project-root/
│
├── main.py                  # FastAPI app
├── streamlit_app.py         # Streamlit UI
├── requirements.txt
│
├── telete/                  # HTML templates (Jinja2)
│   └── index.html
│
├── static/                  # Model and output files
│   ├── mistral_model/       # Mistral 7B model files (config, tokenizer, etc.)
│   └── output/              # Generated QA output (JSON/CSV)
│
├── upload/                  # Uploaded documents (PDF, TXT)
│   └── sample_doc.pdf
│
└── README.md
⚙️ Tech Stack
Model: Mistral 7B Instruct

Backend: FastAPI

Frontend: Streamlit

Template Engine: Jinja2

Document Parsing: PyMuPDF / pdfminer / python-docx

Model Inference: Hugging Face Transformers

🚀 How It Works
Document Upload (via Streamlit):
Users upload .pdf, .txt, or .docx files through the Streamlit UI.

FastAPI Endpoint (/generate):
The backend extracts the text and generates QA pairs using the Mistral 7B model.

QA Generation:
Prompts are constructed and fed to Mistral using the Transformers pipeline.

Output:
The resulting questions and answers are saved in static/output/ as .json or .csv and shown in the UI.

🔧 Setup Instructions
bash
Copy
Edit
git clone https://github.com/yourusername/qa-mistral-fastapi
cd qa-mistral-fastapi

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt
Run FastAPI Backend:

bash
Copy
Edit
uvicorn main:app --reload
Run Streamlit Frontend:

bash
Copy
Edit
streamlit run streamlit_app.py
🧠 Sample Prompt for QA
python
Copy
Edit
prompt = f"""
You're a helpful AI tutor. Read the following content and generate a set of questions with their answers.

Content:
{text_chunk}

Return format:
Q1: <question>
A1: <answer>
Q2: ...
"""
📂 Folders Explained
Folder	Purpose
telete/	Contains Jinja2 HTML templates
static/	Stores model files and QA output
upload/	Stores user-uploaded documents
