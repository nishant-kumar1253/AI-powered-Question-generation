import os
import csv
import json
import aiofiles
import uvicorn
import asyncio
import time
from typing import List, Dict, Optional
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain_community.llms import CTransformers
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Generator API",
    description="API for generating educational content from PDFs using Mistral-7B",
    version="1.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configuration
class Config:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.resolve()
        self.UPLOAD_DIR = self.BASE_DIR / "static/uploads"
        self.OUTPUT_DIR = self.BASE_DIR / "static/output"
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.ensure_directories()
        
    def ensure_directories(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

config = Config()

# Configure static files
static_path = os.path.join(config.BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=config.BASE_DIR / "templates")

# Pydantic models
class MCQItem(BaseModel):
    question: str
    options: List[Dict[str, str]]
    correct_answer: str
    explanation: Optional[str]

class QAItem(BaseModel):
    question: str
    answer: str

# Global LLM instance
llm = None

# Verify model loading during startup
@app.on_event("startup")
async def startup_event():
    global llm
    try:
        logger.info("Loading model...")
        llm = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral",
            context_length = 1,
            config={'max_new_tokens': 4097, 'temperature': 0.3}
        )
        # Test the model
        test_output = await asyncio.to_thread(llm, "Hello")
        logger.info(f"Model test output: {test_output}")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

async def process_pdf(file_path: str):
    try:
        # Add validation
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            logger.info(f"PDF has {len(reader.pages)} pages")
            # Print first 100 chars of first page
            if reader.pages:
                logger.info(f"Sample text: {reader.pages[0].extract_text()[:100]}...")

        # Load and split
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        if not pages:
            raise ValueError("No readable content found in PDF")
        
        full_text = " ".join([
            p.page_content.replace('\n', ' ').strip() 
            for p in pages 
            if p.page_content.strip()
        ])
        
        if not full_text:
            raise ValueError("No extractable text content in PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            length_function=len
        )
        
        documents = text_splitter.split_documents([Document(page_content=full_text)])
        logger.info(f"Split PDF into {len(documents)} chunks")
        
        return documents
    
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise

def generate_mcq_prompt(difficulty: str, question_count: int) -> str:
    return f"""You are an expert educator creating {difficulty} level multiple choice questions.
Generate exactly {question_count} high-quality MCQs with these requirements:
1. Question stem should be clear and concise
2. Provide 4 options (a, b, c, d)
3. Mark the correct answer with [CORRECT]
4. Include a brief explanation

Format each question like this:
Q: [Your question here]
a) [Option 1]
b) [Option 2]
c) [Option 3] [CORRECT]
d) [Option 4]
Explanation: [Your explanation here]

Text to analyze:
{{text}}"""

def generate_qa_prompt(question_count: int) -> str:
    return f"""You are an expert educator creating high-quality question-answer pairs.
Generate exactly {question_count} QA pairs with these requirements:
1. Questions should be clear and directly related to the text
2. Answers should be concise and accurate
3. Both questions and answers should be complete sentences

Format each pair like this:
Q: [Your question here]
A: [Your answer here]

Text to analyze:
{{text}}"""

async def generate_mcqs(documents: List[Document], question_count: int, difficulty: str) -> List[MCQItem]:
    """Generate MCQs from document chunks"""
    try:
        logger.info(f"Generating {question_count} {difficulty} MCQs")
        
        prompt = PromptTemplate(
            template=generate_mcq_prompt(difficulty, question_count),
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        all_mcqs = []
        
        for doc in documents[:5]:  # Limit to first 5 chunks
            try:
                result = await asyncio.to_thread(chain.run, text=doc.page_content)
                mcqs = parse_mcqs(result)
                if mcqs:
                    all_mcqs.extend(mcqs)
                    if len(all_mcqs) >= question_count:
                        break
            except Exception as e:
                logger.warning(f"Error processing chunk: {str(e)}")
                continue
        
        if not all_mcqs:
            raise ValueError("No valid MCQs generated")
        
        return all_mcqs[:question_count]
    
    except Exception as e:
        logger.error(f"MCQ generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")

async def generate_qas(documents: List[Document], question_count: int) -> List[QAItem]:
    """Generate QA pairs from document chunks"""
    try:
        logger.info(f"Generating {question_count} QA pairs")
        
        prompt = PromptTemplate(
            template=generate_qa_prompt(question_count),
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        all_qas = []
        
        for doc in documents[:5]:  # Limit to first 5 chunks
            try:
                result = await asyncio.to_thread(chain.run, text=doc.page_content)
                qas = parse_qas(result)
                if qas:
                    all_qas.extend(qas)
                    if len(all_qas) >= question_count:
                        break
            except Exception as e:
                logger.warning(f"Error processing chunk: {str(e)}")
                continue
        
        if not all_qas:
            raise ValueError("No valid QA pairs generated")
        
        return all_qas[:question_count]
    
    except Exception as e:
        logger.error(f"QA generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"QA generation failed: {str(e)}")

def parse_mcqs(mcq_text: str) -> List[MCQItem]:
    """Parse generated MCQ text into structured format"""
    mcqs = []
    current_q = None
    
    for line in mcq_text.split('\n'):
        line = line.strip()
        
        if line.lower().startswith('q:'):
            if current_q and len(current_q.get('options', [])) == 4:
                mcqs.append(MCQItem(**current_q))
            current_q = {
                'question': line[2:].strip(),
                'options': [],
                'correct_answer': None,
                'explanation': None
            }
        elif line.startswith(('a)', 'b)', 'c)', 'd)')):
            if not current_q:
                continue
                
            parts = line.split(')')
            if len(parts) > 1:
                letter = parts[0].strip().lower()
                text = parts[1].strip()
                is_correct = '[correct]' in text.lower()
                text = text.replace('[CORRECT]', '').replace('[correct]', '').strip()
                
                current_q['options'].append({
                    'letter': letter,
                    'text': text,
                    'correct': is_correct
                })
                
                if is_correct:
                    current_q['correct_answer'] = letter
        elif line.lower().startswith('explanation:') and current_q:
            current_q['explanation'] = line[12:].strip()
    
    if current_q and len(current_q.get('options', [])) == 4:
        mcqs.append(MCQItem(**current_q))
    
    return mcqs

def parse_qas(qa_text: str) -> List[QAItem]:
    """Parse generated QA text into structured format"""
    qas = []
    current_qa = {}
    
    for line in qa_text.split('\n'):
        line = line.strip()
        
        if line.lower().startswith('q:'):
            if current_qa.get('question'):
                if current_qa.get('answer'):
                    qas.append(QAItem(**current_qa))
                else:
                    logger.warning(f"Incomplete QA pair: {current_qa}")
            current_qa = {
                'question': line[2:].strip(),
                'answer': None
            }
        elif line.lower().startswith('a:'):
            if current_qa.get('question'):
                current_qa['answer'] = line[2:].strip()
            else:
                logger.warning(f"Answer without question: {line}")
        
    if current_qa.get('question') and current_qa.get('answer'):
        qas.append(QAItem(**current_qa))
    
    return qas

async def create_output_files(content_type: str, content: List, timestamp: int) -> Dict:
    """Create output files (PDF and CSV)"""
    try:
        output_files = {}
        
        if content_type == "mcq":
            # CSV
            csv_filename = f"mcqs_{timestamp}.csv"
            csv_path = config.OUTPUT_DIR / csv_filename
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct', 'Explanation'])
                for item in content:
                    options = {opt['letter']: opt['text'] for opt in item.options}
                    writer.writerow([
                        item.question,
                        options.get('a', ''),
                        options.get('b', ''),
                        options.get('c', ''),
                        options.get('d', ''),
                        item.correct_answer,
                        item.explanation or ''
                    ])
            
            # PDF
            pdf_filename = f"mcqs_{timestamp}.pdf"
            pdf_path = config.OUTPUT_DIR / pdf_filename
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            y = 750
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, "Generated MCQs")
            y -= 30
            
            for i, item in enumerate(content, 1):
                if y < 100:
                    c.showPage()
                    y = 750
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"{i}. {item.question}")
                y -= 20
                
                for opt in item.options:
                    prefix = "✓ " if opt['correct'] else "  "
                    c.drawString(70, y, f"{prefix}{opt['letter']}) {opt['text']}")
                    y -= 18
                
                if item.explanation:
                    c.setFont("Helvetica-Oblique", 10)
                    c.drawString(70, y, f"Explanation: {item.explanation}")
                    y -= 20
                
                y -= 10
                c.setFont("Helvetica", 12)
            
            c.save()
            
            output_files = {
                "pdf": f"/static/output/{pdf_filename}",
                "csv": f"/static/output/{csv_filename}"
            }
        elif content_type == "qa":
            # CSV for QA pairs
            csv_filename = f"qas_{timestamp}.csv"
            csv_path = config.OUTPUT_DIR / csv_filename
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Question', 'Answer'])
                for item in content:
                    writer.writerow([item.question, item.answer])
            
            # PDF for QA pairs
            pdf_filename = f"qas_{timestamp}.pdf"
            pdf_path = config.OUTPUT_DIR / pdf_filename
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            y = 750
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, "Generated Q&A Pairs")
            y -= 30
            
            for i, item in enumerate(content, 1):
                if y < 100:
                    c.showPage()
                    y = 750
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"{i}. {item.question}")
                y -= 20
                
                c.setFont("Helvetica", 12)
                c.drawString(70, y, f"A: {item.answer}")
                y -= 30
                
                y -= 10
            
            c.save()
            
            output_files = {
                "pdf": f"/static/output/{pdf_filename}",
                "csv": f"/static/output/{csv_filename}"
            }
        
        return output_files
    
    except Exception as e:
        logger.error(f"Output file creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Output creation failed: {str(e)}")

# API Endpoints
@app.post("/upload")
async def upload_file(pdf_file: UploadFile = File(...)):
    """Handle PDF file upload"""
    try:
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted"
            )
        
        filename = f"doc_{int(time.time())}_{pdf_file.filename}"
        filepath = config.UPLOAD_DIR / filename
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(await pdf_file.read())
        
        return {
            "pdf_path": f"/static/uploads/{filename}",
            "message": "File uploaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

@app.post("/generate")
async def generate_content(request: Request):
    """Generate content from PDF"""
    try:
        data = await request.json()
        pdf_path = data.get('pdf_path')
        content_type = data.get('content_type')
        question_count = data.get('question_count', 10)
        difficulty = data.get('difficulty', "medium")

        if not pdf_path or not content_type:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        full_path = config.BASE_DIR / pdf_path.lstrip('/')
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        documents = await process_pdf(str(full_path))
        timestamp = int(time.time())
        
        if content_type == "mcq":
            mcqs = await generate_mcqs(documents, question_count, difficulty)
            output_files = await create_output_files(content_type, mcqs, timestamp)
            return {
                "content": [json.loads(mcq.json()) for mcq in mcqs],
                "output_files": output_files
            }
        elif content_type == "qa":
            qas = await generate_qas(documents, question_count)
            output_files = await create_output_files(content_type, qas, timestamp)
            return {
                "content": [json.loads(qa.json()) for qa in qas],
                "output_files": output_files
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    try:
        file_path = config.OUTPUT_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_frontend(request: Request):
    """Serve the frontend interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )