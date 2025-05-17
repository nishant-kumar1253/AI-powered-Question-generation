# import os
# import csv
# import json
# import aiofiles
# import uvicorn
# import asyncio
# import time
# import re
# from typing import List, Dict, Optional
# from pathlib import Path
# from fastapi.staticfiles import StaticFiles
# from fastapi import FastAPI, Request, File, Form, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.templating import Jinja2Templates
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from PyPDF2 import PdfReader
# from reportlab.pdfgen import canvas
# from reportlab.lib import colors
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer,PageBreak
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# from langchain_community.llms import CTransformers
# from pydantic import BaseModel
# import logging
# from pydantic import BaseModel
# from typing import List, Dict
# from pydantic import validator

# # Add these model definitions at the top of your file, right after the imports
# class MCQOption(BaseModel):
#     letter: str
#     text: str
#     correct: bool

# class MCQItem(BaseModel):
#     question: str
#     options: List[MCQOption]
#     correct_answer: str
#     explanation: str

#     @validator('correct_answer')
#     def validate_correct_answer(cls, v, values):
#         if 'options' not in values:
#             raise ValueError('Options must be provided before correct_answer')
#         option_letters = {opt.letter for opt in values['options']}
#         if v not in option_letters:
#             raise ValueError(f'Correct answer {v} must match one of the option letters')
#         return v

# class QAItem(BaseModel):
#     question: str
#     answer: str

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# app = FastAPI()


# BASE_DIR = Path(__file__).parent


# app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
# templates = Jinja2Templates(directory=BASE_DIR / "templates")

# # Constants
# MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
# ALLOWED_CONTENT_TYPES = ["mcq", "qa"]
# DEFAULT_QUESTION_COUNT = 10
# DEFAULT_DIFFICULTY = "medium"

# # Ensure directories exist
# upload_dir = BASE_DIR / "static/docs"
# output_dir = BASE_DIR / "static/output"
# upload_dir.mkdir(parents=True, exist_ok=True)
# output_dir.mkdir(parents=True, exist_ok=True)

# def load_llm():
#     """Initialize and return the LLM model"""
#     try:
#         logger.info("Loading LLM model...")
#         llm = CTransformers(
#             model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#             model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#             model_type="mistral",
#             config={
#                 'max_new_tokens': 2048,
#                 'temperature': 0.7,
#                 'context_length': 4096,
#                 'gpu_layers': 0
#             }
#         )
#         logger.info("LLM model loaded successfully")
#         return llm
#     except Exception as e:
#         logger.error(f"Failed to load LLM: {str(e)}")
#         raise

# async def save_uploaded_file(file: UploadFile, destination: Path) -> None:
#     """Save uploaded file with chunked reading and size validation"""
#     try:
#         file_size = 0
#         async with aiofiles.open(destination, 'wb') as f:
#             while chunk := await file.read(1024 * 1024):  # 1MB chunks
#                 file_size += len(chunk)
#                 if file_size > MAX_FILE_SIZE:
#                     await f.close()
#                     os.remove(destination)
#                     raise HTTPException(
#                         status_code=413,
#                         detail=f"File exceeds {MAX_FILE_SIZE//(1024*1024)}MB limit"
#                     )
#                 await f.write(chunk)
#         logger.info(f"File saved successfully: {destination}")
#     except Exception as e:
#         logger.error(f"Failed to save file: {str(e)}")
#         raise

# def generate_qa_prompt(question_count: int) -> str:
#     return f"""You are an expert educator creating high-quality question-answer pairs.
# Generate exactly {question_count} QA pairs with these requirements:
# 1. Questions should be clear and directly related to the text
# 2. Answers should be concise and accurate
# 3. Both questions and answers should be complete sentences

# Format each pair like this:
# Q: [Your question here]
# A: [Your answer here]

# Text to analyze:
# {{text}}"""

# def generate_mcq_prompt(difficulty: str, question_count: int) -> str:
#     difficulty_instructions = {
#         "easy": "Focus on basic recall and comprehension",
#         "medium": "Include application and analysis questions",
#         "hard": "Create challenging questions requiring synthesis, evaluation, or complex problem-solving"
#     }
    
#     return f"""Generate exactly {question_count} {difficulty}-level multiple choice questions ({difficulty_instructions[difficulty]}). Follow these STRICT rules:

# 1. FORMAT EACH QUESTION EXACTLY LIKE THIS:
# Q: [Clear, standalone question]
# a) [Option 1]
# b) [Option 2]
# c) [Option 3] [CORRECT]
# d) [Option 4]
# Explanation: [1-2 sentence explanation of why the correct answer is right]

# 2. MUST include:
# - Exactly 4 options per question
# - Exactly one correct option marked with [CORRECT]
# - An explanation for each question
# - Questions that test important concepts from the text

# 3. For {difficulty} questions:
# - {difficulty_instructions[difficulty]}
# - Avoid trivial or obvious questions
# - Include plausible distractors

# Text to analyze:
# {{text}}"""


# def process_pdf(file_path: str) -> List[Document]: #insert into list
#     """Process PDF file and split into chunks"""
#     try:
#         logger.info(f"Processing PDF file: {file_path}")
        
#         loader = PyPDFLoader(file_path)
#         pages = loader.load_and_split()
        
#         if not pages:
#             raise ValueError("No pages found in the PDF document.")
        
#         full_text = " ".join([p.page_content.strip() for p in pages if p.page_content.strip()])
        
#         if not full_text:
#             raise ValueError("No valid text content found in the PDF.")
        
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
        
#         documents = text_splitter.split_documents([Document(page_content=full_text)])
#         logger.info(f"Processed {len(documents)} document chunks")
#         return documents
#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}")
#         raise

# async def generate_output_files(file_path: str, content_type: str, question_count: int, difficulty: str):
#     """Generate output files with requested content type"""
#     try:
#         logger.info(f"Generating {content_type} output from {file_path}")
        
#         documents = await asyncio.to_thread(process_pdf, file_path)
#         llm = load_llm()
#         timestamp = int(time.time())
        
#         if content_type == "mcq":
#             mcqs = await generate_mcqs(documents, llm, question_count, difficulty)
            
#             if not mcqs:
#                 return {"error": "No MCQs generated. The document may not contain enough text content."}
            
#             # Generate output files
#             csv_filename = f"mcqs_{timestamp}.csv"
#             pdf_filename = f"mcqs_{timestamp}.pdf"
            
#             await save_mcq_files(mcqs, csv_filename, pdf_filename)
            
#             return {
#                 "csv_path": f"/static/output/{csv_filename}",
#                 "pdf_path": f"/static/output/{pdf_filename}",
#                 "mcqs": mcqs
#             }
#         else:
#             qa_pairs = await generate_qa_pairs(documents, llm, question_count)  # Added question_count here
            
#             if not qa_pairs:
#                 return {"error": "No Q&A pairs generated. The document may not contain enough text content."}
            
#             # Generate output files
#             csv_filename = f"qa_{timestamp}.csv"
#             pdf_filename = f"qa_{timestamp}.pdf"
            
#             await save_qa_files(qa_pairs, csv_filename, pdf_filename)
            
#             return {
#                 "csv_path": f"/static/output/{csv_filename}",
#                 "pdf_path": f"/static/output/{pdf_filename}",
#                 "qa_pairs": qa_pairs
#             }
#     except Exception as e:
#         logger.error(f"Error generating output: {str(e)}")
#         return {"error": str(e)}


# async def generate_mcqs(documents: List[Document], llm: CTransformers, question_count: int, difficulty: str) -> List[Dict]:
#     """Generate MCQs from document chunks"""
#     try:
#         logger.info(f"Generating {question_count} {difficulty} MCQs")
        
#         prompt_template = generate_mcq_prompt(difficulty, question_count)
#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=["text"]
#         )
        
#         chain = LLMChain(llm=llm, prompt=prompt)
#         all_mcqs = []
#         attempts = 0
#         max_attempts = min(10, len(documents))  # Try up to 10 chunks or all available
        
#         while len(all_mcqs) < question_count and attempts < max_attempts:
#             doc = documents[attempts]
#             try:
#                 result = await asyncio.to_thread(chain.run, text=doc.page_content)
#                 mcqs = parse_mcqs(result)
#                 if mcqs:
#                     all_mcqs.extend(mcqs)
#                 attempts += 1
#             except Exception as e:
#                 logger.warning(f"Error processing chunk {attempts}: {str(e)}")
#                 attempts += 1
#                 continue
        
#         if not all_mcqs:
#             raise ValueError("No valid MCQs generated")
        
#         # Ensure we return exactly the requested number
#         return [mcq.dict() for mcq in all_mcqs[:question_count]]
    
#     except Exception as e:
#         logger.error(f"MCQ generation failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")

# def parse_mcqs(mcq_text: str) -> List[MCQItem]:
#     mcqs = []
#     current_q = None
    
#     for line in mcq_text.split('\n'):
#         line = line.strip()
        
#         # More flexible question detection
#         if line.lower().startswith(('q:', 'question:', 'mcq:')):
#             if current_q and len(current_q.get('options', [])) >= 1:  # Allow partial saves
#                 try:
#                     mcqs.append(MCQItem(**current_q))
#                 except Exception:
#                     continue
#             current_q = {
#                 'question': line.split(':', 1)[1].strip(),
#                 'options': [],
#                 'correct_answer': None,
#                 'explanation': None
#             }
        
#         # Flexible option detection (a), b., c:, etc.)
#         elif re.match(r'^[a-d][).:]', line.lower()):
#             if not current_q:
#                 continue
#             letter = line[0].lower()
#             text = re.sub(r'\[correct\]|\(correct\)', '', line[2:].strip(), flags=re.IGNORECASE)
#             is_correct = 'correct' in line.lower()
            
#             current_q['options'].append({
#                 'letter': letter,
#                 'text': text.strip(),
#                 'correct': is_correct
#             })
            
#             if is_correct:
#                 current_q['correct_answer'] = letter
        
#         # Flexible explanation detection
#         elif line.lower().startswith(('explanation:', 'reason:')):
#             if current_q:
#                 current_q['explanation'] = line.split(':', 1)[1].strip()
    
#     # Add the last question if valid
#     if current_q and len(current_q.get('options', [])) >= 1:
#         try:
#             mcqs.append(MCQItem(**current_q))
#         except Exception:
#             pass
    
#     return mcqs

# def validate_current_question(current_q: dict) -> bool:
#     """Validate that a question has all required components"""
#     if not current_q.get('question'):
#         return False
        
#     if len(current_q.get('options', [])) != 4:
#         return False
        
#     if not current_q.get('correct_answer'):
#         return False
        
#     # Verify correct answer matches one of the options
#     correct_letters = {opt['letter'] for opt in current_q['options'] if opt['correct']}
#     return current_q['correct_answer'] in correct_letters

# def create_mcq_item(current_q: dict) -> MCQItem:
#     """Create MCQItem with proper validation"""
#     # Ensure options are properly ordered a-d
#     ordered_options = sorted(current_q['options'], key=lambda x: x['letter'])
#     current_q['options'] = ordered_options
    
#     return MCQItem(
#         question=current_q['question'],
#         options=[MCQOption(**opt) for opt in current_q['options']],
#         correct_answer=current_q['correct_answer'],
#         explanation=current_q.get('explanation', '')
#     )
# def parse_qas(qa_text: str) -> List[QAItem]:
#     """Parse generated QA text into structured format"""
#     qas = []
#     current_qa = {}
    
#     for line in qa_text.split('\n'):
#         line = line.strip()
        
#         if line.lower().startswith('q:'):
#             if current_qa.get('question'):
#                 if current_qa.get('answer'):
#                     qas.append(QAItem(**current_qa))
#                 else:
#                     logger.warning(f"Incomplete QA pair: {current_qa}")
#             current_qa = {
#                 'question': line[2:].strip(),
#                 'answer': None
#             }
#         elif line.lower().startswith('a:'):
#             if current_qa.get('question'):
#                 current_qa['answer'] = line[2:].strip()
#             else:
#                 logger.warning(f"Answer without question: {line}")
        
#     if current_qa.get('question') and current_qa.get('answer'):
#         qas.append(QAItem(**current_qa))
    
#     return qas

# async def generate_qa_pairs(documents: List[Document], llm: CTransformers, question_count: int) -> List[Dict]:
#     """Generate QA pairs from document chunks"""
#     try:
#         logger.info(f"Generating {question_count} QA pairs")
        
#         prompt_template = generate_qa_prompt(question_count)
#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=["text"]
#         )
        
#         chain = LLMChain(llm=llm, prompt=prompt)
#         all_qas = []
        
#         for doc in documents[:5]:  # Limit to first 5 chunks
#             try:
#                 result = await asyncio.to_thread(chain.run, text=doc.page_content)
#                 qas = parse_qas(result)
#                 if qas:
#                     all_qas.extend(qas)
#                     if len(all_qas) >= question_count:
#                         break
#             except Exception as e:
#                 logger.warning(f"Error processing chunk: {str(e)}")
#                 continue
        
#         if not all_qas:
#             raise ValueError("No valid QA pairs generated")
        
#         # Convert to dict for JSON serialization
#         return [qa.dict() for qa in all_qas[:question_count]]
    
#     except Exception as e:
#         logger.error(f"QA generation failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"QA generation failed: {str(e)}")

# async def save_mcq_files(mcqs: List[Dict], csv_filename: str, pdf_filename: str) -> None:
#     """Save MCQs to CSV and PDF files with guaranteed text wrapping"""
#     csv_path = output_dir / csv_filename
#     pdf_path = output_dir / pdf_filename
    
#     # Save to CSV (existing code remains the same)
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Explanation'])
        
#         for mcq in mcqs:
#             options = {opt['letter']: opt['text'] for opt in mcq['options']}
#             writer.writerow([
#                 mcq['question'],
#                 options.get('a', ''),
#                 options.get('b', ''),
#                 options.get('c', ''),
#                 options.get('d', ''),
#                 mcq['correct_answer'],
#                 mcq['explanation']
#             ])
    
#     # Create PDF with proper text wrapping
#     doc = SimpleDocTemplate(
#         str(pdf_path),
#         pagesize=letter,
#         leftMargin=0.75*inch,
#         rightMargin=0.75*inch,
#         topMargin=0.75*inch,
#         bottomMargin=0.75*inch
#     )
    
#     # Define custom styles with proper wrapping
#     styles = getSampleStyleSheet()
    
#     # Main question style
#     question_style = ParagraphStyle(
#         'QuestionStyle',
#         parent=styles['Normal'],
#         fontName='Helvetica-Bold',
#         fontSize=12,
#         leading=14,
#         spaceAfter=6,
#         wordWrap='LTR'  # Ensure left-to-right word wrapping
#     )
    
#     # Option style with hanging indent
#     option_style = ParagraphStyle(
#         'OptionStyle',
#         parent=styles['Normal'],
#         fontSize=10,
#         leftIndent=18,
#         spaceAfter=2,
#         wordWrap='LTR'
#     )
    
#     # Correct answer style
#     correct_style = ParagraphStyle(
#         'CorrectStyle',
#         parent=styles['Normal'],
#         fontName='Helvetica-Bold',
#         textColor=colors.green,
#         fontSize=10,
#         spaceAfter=6,
#         wordWrap='LTR'
#     )
    
#     # Explanation style
#     explanation_style = ParagraphStyle(
#         'ExplanationStyle',
#         parent=styles['Normal'],
#         fontName='Helvetica-Oblique',
#         textColor=colors.blue,
#         fontSize=10,
#         spaceAfter=12,
#         wordWrap='LTR'
#     )
    
#     # Build the story (content)
#     story = []
    
#     for i, mcq in enumerate(mcqs, 1):
#         # Add question with proper wrapping
#         question_text = f"<b>Q{i}:</b> {mcq['question']}"
#         story.append(Paragraph(question_text, question_style))
        
#         # Add options with proper wrapping
#         for opt in mcq['options']:
#             option_text = f"{opt['letter']}) {opt['text']}"
#             if opt['correct']:
#                 option_text = f"<font color=green><b>{option_text}</b></font>"
#             story.append(Paragraph(option_text, option_style))
        
#         # Add correct answer
#         correct_text = f"<b>Correct Answer:</b> {mcq['correct_answer']}"
#         story.append(Paragraph(correct_text, correct_style))
        
#         # Add explanation
#         if mcq['explanation']:
#             explanation_text = f"<i>Explanation:</i> {mcq['explanation']}"
#             story.append(Paragraph(explanation_text, explanation_style))
        
#         # Add space between questions
#         story.append(Spacer(1, 0.2*inch))
    
#     # Build the PDF document
#     doc.build(story)
   

# async def save_qa_files(qa_pairs: List[Dict], csv_filename: str, pdf_filename: str) -> None:
#     """Save Q&A pairs to CSV and PDF files with proper text wrapping"""
#     csv_path = output_dir / csv_filename
#     pdf_path = output_dir / pdf_filename
    
#     # Save to CSV (existing code remains the same)
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Question', 'Answer'])
#         for qa in qa_pairs:
#             writer.writerow([qa['question'], qa['answer']])
    
#     # Create PDF with proper text wrapping
#     doc = SimpleDocTemplate(
#         str(pdf_path),
#         pagesize=letter,
#         leftMargin=0.75*inch,
#         rightMargin=0.75*inch,
#         topMargin=0.75*inch,
#         bottomMargin=0.75*inch
#     )
    
#     # Define custom styles
#     styles = getSampleStyleSheet()
    
#     # Question style
#     question_style = ParagraphStyle(
#         'QuestionStyle',
#         parent=styles['Normal'],
#         fontName='Helvetica-Bold',
#         fontSize=12,
#         leading=14,
#         spaceAfter=6,
#         textColor=colors.navy,
#         wordWrap='LTR'
#     )
    
#     # Answer style
#     answer_style = ParagraphStyle(
#         'AnswerStyle',
#         parent=styles['Normal'],
#         fontSize=11,
#         leading=13,
#         leftIndent=20,
#         spaceAfter=12,
#         textColor=colors.darkgreen,
#         wordWrap='LTR'
#     )
    
#     # Build the content
#     story = []
    
#     for i, qa in enumerate(qa_pairs, 1):
#         # Add question with numbering
#         question_text = f"<b>Q{i}:</b> {qa['question']}"
#         story.append(Paragraph(question_text, question_style))
        
#         # Add answer with indentation
#         answer_text = f"<b>Answer:</b> {qa['answer']}"
#         story.append(Paragraph(answer_text, answer_style))
        
#         # Add space between Q&A pairs
#         story.append(Spacer(1, 0.2*inch))
        
#         # Add page break if less than 2 inches remaining
#         if i < len(qa_pairs) and (i % 5 == 0):
#             story.append(PageBreak())
    
#     # Build the PDF document
#     doc.build(story)

    



# @app.get("/")
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/upload")
# async def upload_pdf(
#     pdf_file: UploadFile = File(...),
#     content_type: str = Form(...),
#     question_count: int = Form(DEFAULT_QUESTION_COUNT),
#     difficulty: str = Form(DEFAULT_DIFFICULTY)
# ):
#     try:
#         if content_type not in ALLOWED_CONTENT_TYPES:
#             raise HTTPException(status_code=400, detail=f"Invalid content type. Allowed types: {ALLOWED_CONTENT_TYPES}")
        
#         if not pdf_file.filename.lower().endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="Only PDF files are allowed")

#         timestamp = int(time.time())
#         filename = f"doc_{timestamp}.pdf"
#         pdf_path = upload_dir / filename
        
#         await save_uploaded_file(pdf_file, pdf_path)
        
#         return JSONResponse({
#             "status": "success",
#             "pdf_path": f"/static/docs/{filename}",
#             "content_type": content_type,
#             "question_count": question_count,
#             "difficulty": difficulty
#         })
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Upload failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="File upload failed. Please try again.")

# # @app.post("/analyze")
# # async def analyze_pdf(
# #     pdf_path: str = Form(...),
# #     content_type: str = Form(...),
# #     question_count: int = Form(DEFAULT_QUESTION_COUNT),
# #     difficulty: str = Form(DEFAULT_DIFFICULTY)
# # ):
# #     try:
# #         if not pdf_path.startswith("/static/docs/"):
# #             raise HTTPException(status_code=400, detail="Invalid file path format")
        
# #         full_path = BASE_DIR / pdf_path.lstrip("/")
        
# #         if not full_path.exists():
# #             raise HTTPException(status_code=404, detail="File not found")
        
# #         output = await generate_output_files(
# #             str(full_path),
# #             content_type,
# #             question_count,
# #             difficulty
# #         )
        
# #         if "error" in output:
# #             raise HTTPException(status_code=400, detail=output["error"])
        
# #         return JSONResponse({
# #             "status": "success",
# #             "output_files": {
# #                 "csv_path": output["csv_path"],  # Use the path from output
# #                 "pdf_path": output["pdf_path"],  # Use the path from output
# #             },
# #             "content": output.get("mcqs", output.get("qa_pairs", []))
# #         })
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         logger.error(f"Analysis failed: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# @app.post("/analyze")
# async def analyze_pdf(
#     pdf_path: str = Form(...),
#     content_type: str = Form(...),
#     question_count: int = Form(DEFAULT_QUESTION_COUNT),
#     difficulty: str = Form(DEFAULT_DIFFICULTY)
# ):
#     try:
#         if not pdf_path.startswith("/static/docs/"):
#             raise HTTPException(status_code=400, detail="Invalid file path format")
        
#         full_path = BASE_DIR / pdf_path.lstrip("/")
        
#         if not full_path.exists():
#             raise HTTPException(status_code=404, detail="File not found")
        
#         output = await generate_output_files(
#             str(full_path),
#             content_type,
#             question_count,
#             difficulty
#         )
        
#         if "error" in output:
#             raise HTTPException(status_code=400, detail=output["error"])
        
#         return JSONResponse({
#             "status": "success",
#             "output_files": {
#                 "csv": output["csv_path"],  
#                 "pdf": output["pdf_path"],  
#             },
#             "content": output.get("mcqs", output.get("qa_pairs", []))
#         })
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Analysis failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
# @app.get("/download/{filename:path}")
# async def download_file(filename: str):
#     try:
#         file_path = output_dir / filename
        
#         # Security check - prevent directory traversal
#         if not file_path.resolve().parent.samefile(output_dir.resolve()):
#             raise HTTPException(status_code=403, detail="Access denied")
            
#         if not file_path.exists():
#             raise HTTPException(status_code=404, detail="File not found")
            
#         return FileResponse(
#             file_path,
#             filename=filename,
#             media_type="application/octet-stream"
#         )
#     except Exception as e:
#         logger.error(f"Download failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="File download failed")

# if __name__ == "__main__":
#     uvicorn.run(
#         "app:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )/

import os
import csv
import json
import aiofiles
import uvicorn
import asyncio
import time
import re
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
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from langchain_community.llms import CTransformers
from pydantic import BaseModel
import logging
from pydantic import validator, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from html import escape


# Model definitions
class MCQOption(BaseModel):
    letter: str = Field(..., pattern="^[a-d]$")
    text: str
    correct: bool

class MCQItem(BaseModel):
    question: str
    options: List[MCQOption]
    correct_answer: str = Field(..., pattern="^[a-d]$")
    explanation: Optional[str] = None

    @validator('correct_answer')
    def validate_correct_answer(cls, v, values):
        if 'options' not in values:
            raise ValueError('Options must be provided before correct_answer')
        option_letters = {opt.letter for opt in values['options']}
        if v not in option_letters:
            raise ValueError(f'Correct answer {v} must match one of the option letters')
        return v

    @validator('options')
    def validate_options(cls, v):
        if len(v) != 4:
            raise ValueError('Exactly 4 options are required')
        letters = {opt.letter for opt in v}
        if letters != {'a', 'b', 'c', 'd'}:
            raise ValueError('Options must have letters a, b, c, d')
        if sum(1 for opt in v if opt.correct) != 1:
            raise ValueError('Exactly one option must be correct')
        return v

class QAItem(BaseModel):
    question: str
    answer: str

class PDFProcessRequest(BaseModel):
    difficulty: str
    question_count: int

class PaperSection(BaseModel):
    name: str
    instructions: str
    marksPerQuestion: int
    mcqCount: int
    descCount: int

class GeneratePaperRequest(BaseModel):
    institution: str
    examType: str
    year: str
    course: str
    time: str
    marks: int
    instructions: str
    sections: List[PaperSection]
    autoGeneratedQuestions: List[dict] = []

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
BASE_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_CONTENT_TYPES = ["mcq", "qa"]
DEFAULT_QUESTION_COUNT = 10
DEFAULT_DIFFICULTY = "medium"
MAX_RETRY_ATTEMPTS = 3

# Ensure directories exist
upload_dir = BASE_DIR / "static/docs"
output_dir = BASE_DIR / "static/output"
upload_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Add CORS middleware (right after creating the FastAPI app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_llm():
    """Initialize and return the LLM model"""
    try:
        logger.info("Loading LLM model...")
        llm = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral",
            config={
                'max_new_tokens': 2048,
                'temperature': 0.7,
                'context_length': 4096,
                'gpu_layers': 0
            }
        )
        logger.info("LLM model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLM: {str(e)}")
        raise

async def save_uploaded_file(file: UploadFile, destination: Path) -> None:
    """Save uploaded file with chunked reading and size validation"""
    try:
        file_size = 0
        async with aiofiles.open(destination, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    await f.close()
                    os.remove(destination)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_FILE_SIZE//(1024*1024)}MB limit"
                    )
                await f.write(chunk)
        logger.info(f"File saved successfully: {destination}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise

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

def generate_mcq_prompt(difficulty: str, question_count: int) -> str:
    difficulty_instructions = {
        "easy": "Focus on basic recall and comprehension",
        "medium": "Include application and analysis questions",
        "hard": "Create challenging questions requiring synthesis, evaluation, or complex problem-solving"
    }
    
    return f"""Generate exactly {question_count} high-quality multiple choice questions from the provided text. Follow these rules STRICTLY:

1. FORMAT FOR EACH QUESTION (MUST FOLLOW THIS EXACT STRUCTURE):
---
Q: [Your question here]
a) [Option 1] [CORRECT]
b) [Option 2]
c) [Option 3]
d) [Option 4]
Explanation: [Brief explanation of why the correct answer is right]
---

2. REQUIREMENTS:
- Each question must test important concepts from the text
- Questions should be self-contained and clear
- Provide exactly 4 options (a-d) for each question
- Mark exactly one correct answer per question with [CORRECT]
- Include a concise explanation for each question
- Difficulty level: {difficulty} ({difficulty_instructions[difficulty]})
- Generate questions from diverse parts of the text

3. EXAMPLE OUTPUT:
---
Q: What is the primary function of the mitochondria?
a) Protein synthesis
b) Energy production [CORRECT]
c) DNA replication
d) Waste removal
Explanation: Mitochondria are known as the powerhouse of the cell, responsible for energy production through ATP synthesis.
---

Text to analyze:
{{text}}"""

def process_pdf(file_path: str) -> List[Document]:
    """Process PDF file and split into chunks"""
    try:
        logger.info(f"Processing PDF file: {file_path}")
        
        # First try with PyPDFLoader
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
        except Exception as e:
            logger.warning(f"PyPDFLoader failed, trying PyPDF2 as fallback: {str(e)}")
            # Fallback to PyPDF2
            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(Document(page_content=text))
        
        if not pages:
            raise ValueError("No pages or text content found in the PDF document.")
        
        full_text = " ".join([p.page_content.strip() for p in pages if p.page_content.strip()])
        
        if not full_text:
            raise ValueError("No valid text content could be extracted from the PDF.")
        
        if len(full_text) < 100:
            logger.warning(f"Extracted text seems very short ({len(full_text)} characters)")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        documents = text_splitter.split_documents([Document(page_content=full_text)])
        logger.info(f"Processed {len(documents)} document chunks")
        return documents
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise

def parse_mcqs(mcq_text: str) -> List[MCQItem]:
    """Robust MCQ parser with multiple pattern matching strategies"""
    mcqs = []
    patterns = [
        # Standard pattern
        r'Q:\s*(.*?)\s*a\)\s*(.*?)(?:\s*\[CORRECT\])?\s*b\)\s*(.*?)\s*c\)\s*(.*?)\s*d\)\s*(.*?)\s*Explanation:\s*(.*)',
        # Variant pattern
        r'Question\s*\d*:?\s*(.*?)\s*\(?a\)\s*(.*?)(?:\s*\[CORRECT\])?\s*\(?b\)\s*(.*?)\s*\(?c\)\s*(.*?)\s*\(?d\)\s*(.*?)\s*Explanation:\s*(.*)'
    ]
    
    # First try structured parsing
    for pattern in patterns:
        matches = re.finditer(pattern, mcq_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                question = match.group(1).strip()
                options = [
                    {'letter': 'a', 'text': match.group(2).strip(), 'correct': '[CORRECT]' in match.group(2).upper()},
                    {'letter': 'b', 'text': match.group(3).strip(), 'correct': False},
                    {'letter': 'c', 'text': match.group(4).strip(), 'correct': False},
                    {'letter': 'd', 'text': match.group(5).strip(), 'correct': False}
                ]
                
                # If no correct option marked, assume first option is correct
                if not any(opt['correct'] for opt in options):
                    options[0]['correct'] = True
                
                correct_answer = next(opt['letter'] for opt in options if opt['correct'])
                explanation = match.group(6).strip() if len(match.groups()) > 5 else None
                
                mcqs.append(MCQItem(
                    question=question,
                    options=[MCQOption(**opt) for opt in options],
                    correct_answer=correct_answer,
                    explanation=explanation
                ))
            except Exception as e:
                logger.warning(f"Failed to parse MCQ with pattern {pattern}: {str(e)}")
                continue
    
    # If structured parsing failed, try line-by-line parsing
    if not mcqs:
        current_q = {}
        for line in mcq_text.split('\n'):
            line = line.strip()
            
            # Question detection
            if re.match(r'^(Q:|Question\s*\d*:)\s*', line, re.IGNORECASE):
                if current_q and validate_current_question(current_q):
                    try:
                        mcqs.append(create_mcq_item(current_q))
                    except Exception as e:
                        logger.warning(f"Failed to create MCQ from current_q: {str(e)}")
                current_q = {
                    'question': re.sub(r'^(Q:|Question\s*\d*:)\s*', '', line, flags=re.IGNORECASE).strip(),
                    'options': [],
                    'correct_answer': None,
                    'explanation': None
                }
            
            # Option detection
            elif re.match(r'^[a-d][).]\s*', line, re.IGNORECASE):
                if not current_q:
                    continue
                letter = line[0].lower()
                text = re.sub(r'^[a-d][).]\s*', '', line).strip()
                is_correct = '[CORRECT]' in text.upper()
                text = re.sub(r'\[CORRECT\]', '', text, flags=re.IGNORECASE).strip()
                
                current_q['options'].append({
                    'letter': letter,
                    'text': text,
                    'correct': is_correct
                })
                
                if is_correct:
                    current_q['correct_answer'] = letter
            
            # Explanation detection
            elif re.match(r'^Explanation:\s*', line, re.IGNORECASE):
                if current_q:
                    current_q['explanation'] = re.sub(r'^Explanation:\s*', '', line, flags=re.IGNORECASE).strip()
        
        # Add the last question if valid
        if current_q and validate_current_question(current_q):
            try:
                mcqs.append(create_mcq_item(current_q))
            except Exception as e:
                logger.warning(f"Failed to create final MCQ: {str(e)}")
    
    return mcqs

def validate_current_question(current_q: dict) -> bool:
    """Validate that current question has minimum required fields"""
    if not current_q.get('question'):
        return False
        
    if len(current_q.get('options', [])) < 2:  # At least 2 options
        return False
        
    # Check that at least one option is marked correct
    if not any(opt.get('correct') for opt in current_q.get('options', [])):
        return False
        
    return True

def create_mcq_item(current_q: dict) -> MCQItem:
    """Create MCQItem with proper validation and ordering"""
    try:
        # Ensure options are properly ordered a-d
        if 'options' not in current_q:
            raise ValueError("Missing options")
            
        ordered_options = sorted(
            [opt for opt in current_q['options'] if 'letter' in opt and 'text' in opt],
            key=lambda x: x['letter']
        )
        
        if len(ordered_options) < 2:
            raise ValueError("Not enough valid options")
            
        # Find correct answer if not specified
        if not current_q.get('correct_answer'):
            correct_options = [opt['letter'] for opt in ordered_options if opt.get('correct')]
            if correct_options:
                current_q['correct_answer'] = correct_options[0]
            else:
                raise ValueError("No correct option specified")
        
        return MCQItem(
            question=current_q.get('question', ''),
            options=[MCQOption(**opt) for opt in ordered_options],
            correct_answer=current_q.get('correct_answer', ''),
            explanation=current_q.get('explanation')
        )
    except Exception as e:
        logger.error(f"Error creating MCQItem: {str(e)}")
        raise

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

async def generate_mcqs(documents: List[Document], llm: CTransformers, question_count: int, difficulty: str) -> List[Dict]:
    """Generate MCQs from document chunks with retry logic"""
    all_mcqs = []
    retry_attempts = 0
    
    while len(all_mcqs) < question_count and retry_attempts < MAX_RETRY_ATTEMPTS:
        try:
            logger.info(f"Attempt {retry_attempts + 1} to generate MCQs")
            
            prompt_template = generate_mcq_prompt(difficulty, question_count - len(all_mcqs))
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["text"]
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            
            for doc in documents:
                try:
                    result = await asyncio.to_thread(chain.run, text=doc.page_content)
                    logger.debug(f"Raw LLM output:\n{result}")
                    
                    mcqs = parse_mcqs(result)
                    logger.info(f"Parsed {len(mcqs)} MCQs from chunk")
                    
                    if mcqs:
                        all_mcqs.extend(mcq.dict() for mcq in mcqs)
                        if len(all_mcqs) >= question_count:
                            break
                
                except Exception as e:
                    logger.warning(f"Error processing chunk: {str(e)}")
                    continue
            
            if not all_mcqs and retry_attempts < MAX_RETRY_ATTEMPTS - 1:
                retry_attempts += 1
                logger.info(f"Retrying MCQ generation (attempt {retry_attempts})")
                continue
            
        except Exception as e:
            logger.error(f"MCQ generation error: {str(e)}")
            if retry_attempts < MAX_RETRY_ATTEMPTS - 1:
                retry_attempts += 1
                continue
            raise
    
    if not all_mcqs:
        error_msg = "No valid MCQs generated after all attempts. Possible reasons:\n"
        error_msg += "1. The document may not contain enough meaningful text\n"
        error_msg += "2. The content may not be suitable for MCQ generation\n"
        error_msg += "3. The LLM may not be following the required format\n"
        error_msg += "Please try with a different document or adjust the parameters."
        raise HTTPException(status_code=400, detail=error_msg)
    
    return all_mcqs[:question_count]

async def generate_qa_pairs(documents: List[Document], llm: CTransformers, question_count: int) -> List[Dict]:
    """Generate QA pairs from document chunks"""
    try:
        logger.info(f"Generating {question_count} QA pairs")
        
        prompt_template = generate_qa_prompt(question_count)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        all_qas = []
        
        for doc in documents[:5]:  # Limit to first 5 chunks
            try:
                result = await asyncio.to_thread(chain.run, text=doc.page_content)
                qas = parse_qas(result)
                if qas:
                    all_qas.extend(qa.dict() for qa in qas)
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

async def save_mcq_files(mcqs: List[Dict], csv_filename: str, pdf_filename: str) -> None:
    """Save MCQs to CSV and PDF files with proper formatting"""
    csv_path = output_dir / csv_filename
    pdf_path = output_dir / pdf_filename
    
    # Save to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Explanation'])
        
        for mcq in mcqs:
            options = {opt['letter']: opt['text'] for opt in mcq['options']}
            writer.writerow([
                mcq['question'],
                options.get('a', ''),
                options.get('b', ''),
                options.get('c', ''),
                options.get('d', ''),
                mcq['correct_answer'],
                mcq['explanation'] or ''
            ])
    
    # Create PDF with proper text wrapping
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * 72,
        rightMargin=0.75 * 72,
        topMargin=0.75 * 72,
        bottomMargin=0.75 * 72
    )
    
    # Define custom styles
    styles = getSampleStyleSheet()
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        spaceAfter=6,
        wordWrap='LTR'
    )
    
    option_style = ParagraphStyle(
        'OptionStyle',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=18,
        spaceAfter=2,
        wordWrap='LTR'
    )
    
    correct_style = ParagraphStyle(
        'CorrectStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        textColor=colors.green,
        fontSize=10,
        spaceAfter=6,
        wordWrap='LTR'
    )
    
    explanation_style = ParagraphStyle(
        'ExplanationStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        textColor=colors.blue,
        fontSize=10,
        spaceAfter=12,
        wordWrap='LTR'
    )
    
    # Build the content
    story = []
    
    for i, mcq in enumerate(mcqs, 1):
        # Add question
        question_text = f"<b>Q{i}:</b> {mcq['question']}"
        story.append(Paragraph(question_text, question_style))
        
        # Add options
        for opt in mcq['options']:
            option_text = f"{opt['letter']}) {opt['text']}"
            if opt['correct']:
                option_text = f"<font color=green><b>{option_text}</b></font>"
            story.append(Paragraph(option_text, option_style))
        
        # Add correct answer
        correct_text = f"<b>Correct Answer:</b> {mcq['correct_answer']}"
        story.append(Paragraph(correct_text, correct_style))
        
        # Add explanation if exists
        if mcq['explanation']:
            explanation_text = f"<i>Explanation:</i> {mcq['explanation']}"
            story.append(Paragraph(explanation_text, explanation_style))
        
        # Add space between questions
        story.append(Spacer(1, 12))
    
    # Build the PDF document
    doc.build(story)

async def save_qa_files(qa_pairs: List[Dict], csv_filename: str, pdf_filename: str) -> None:
    """Save Q&A pairs to CSV and PDF files"""
    csv_path = output_dir / csv_filename
    pdf_path = output_dir / pdf_filename
    
    # Save to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])
        for qa in qa_pairs:
            writer.writerow([qa['question'], qa['answer']])
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * 72,
        rightMargin=0.75 * 72,
        topMargin=0.75 * 72,
        bottomMargin=0.75 * 72
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        spaceAfter=6,
        textColor=colors.navy,
        wordWrap='LTR'
    )
    
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=13,
        leftIndent=20,
        spaceAfter=12,
        textColor=colors.darkgreen,
        wordWrap='LTR'
    )
    
    # Build content
    story = []
    
    for i, qa in enumerate(qa_pairs, 1):
        # Add question
        question_text = f"<b>Q{i}:</b> {qa['question']}"
        story.append(Paragraph(question_text, question_style))
        
        # Add answer
        answer_text = f"<b>Answer:</b> {qa['answer']}"
        story.append(Paragraph(answer_text, answer_style))
        
        # Add space between Q&A pairs
        story.append(Spacer(1, 12))
        
        # Add page break if needed
        if i < len(qa_pairs) and (i % 5 == 0):
            story.append(PageBreak())
    
    # Build PDF
    doc.build(story)

async def generate_output_files(file_path: str, content_type: str, question_count: int, difficulty: str):
    """Generate output files with requested content type"""
    try:
        logger.info(f"Generating {content_type} output from {file_path}")
        
        documents = await asyncio.to_thread(process_pdf, file_path)
        llm = load_llm()
        timestamp = int(time.time())
        
        if content_type == "mcq":
            mcqs = await generate_mcqs(documents, llm, question_count, difficulty)
            
            if not mcqs:
                return {"error": "No MCQs generated. The document may not contain enough text content."}
            
            # Generate output files
            csv_filename = f"mcqs_{timestamp}.csv"
            pdf_filename = f"mcqs_{timestamp}.pdf"
            
            await save_mcq_files(mcqs, csv_filename, pdf_filename)
            
            return {
                "csv_path": f"/static/output/{csv_filename}",
                "pdf_path": f"/static/output/{pdf_filename}",
                "mcqs": mcqs
            }
        else:
            qa_pairs = await generate_qa_pairs(documents, llm, question_count)
            
            if not qa_pairs:
                return {"error": "No Q&A pairs generated. The document may not contain enough text content."}
            
            # Generate output files
            csv_filename = f"qa_{timestamp}.csv"
            pdf_filename = f"qa_{timestamp}.pdf"
            
            await save_qa_files(qa_pairs, csv_filename, pdf_filename)
            
            return {
                "csv_path": f"/static/output/{csv_filename}",
                "pdf_path": f"/static/output/{pdf_filename}",
                "qa_pairs": qa_pairs
            }
    except Exception as e:
        logger.error(f"Error generating output: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.get('/favicon.ico', include_in_schema=False)
# async def favicon():
#     return FileResponse('static/favicon.ico')

@app.post("/upload")
async def upload_pdf(
    pdf_file: UploadFile = File(...),
    content_type: str = Form(...),
    question_count: int = Form(DEFAULT_QUESTION_COUNT),
    difficulty: str = Form(DEFAULT_DIFFICULTY)
):
    try:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid content type. Allowed types: {ALLOWED_CONTENT_TYPES}")
        
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        timestamp = int(time.time())
        filename = f"doc_{timestamp}.pdf"
        pdf_path = upload_dir / filename
        
        await save_uploaded_file(pdf_file, pdf_path)
        
        return JSONResponse({
            "status": "success",
            "pdf_path": f"/static/docs/{filename}",
            "content_type": content_type,
            "question_count": question_count,
            "difficulty": difficulty
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="File upload failed. Please try again.")
    
@app.post("/process-pdf")
async def process_pdf_endpoint(
    pdf_file: UploadFile = File(...),
    difficulty: str = Form(...),
    question_count: int = Form(...),
    content_type: str = Form("mcq")
):
    try:
        # Validate inputs
        question_count = int(question_count)
        if question_count < 1 or question_count > 50:
            raise HTTPException(status_code=400, detail="Question count must be between 1-50")
            
        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty level")
            
        if content_type not in ["mcq", "qa"]:
            raise HTTPException(status_code=400, detail="Invalid content type")

        # Save file
        timestamp = int(time.time())
        filename = f"doc_{timestamp}.pdf"
        pdf_path = upload_dir / filename
        await save_uploaded_file(pdf_file, pdf_path)

        # Process PDF
        documents = await asyncio.to_thread(process_pdf, str(pdf_path))
        llm = load_llm()
        
        # Generate questions
        if content_type == "mcq":
            prompt_template = generate_mcq_prompt(difficulty, question_count)
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            runnable = prompt | llm
            
            questions = []
            for doc in documents[:3]:  # Limit to first 3 chunks to avoid timeout
                try:
                    result = await runnable.ainvoke({"text": doc.page_content})
                    mcqs = parse_mcqs(result)
                    questions.extend(mcq.dict() for mcq in mcqs)
                    if len(questions) >= question_count:
                        break
                except Exception as e:
                    logger.warning(f"Error processing chunk: {str(e)}")
                    continue
                    
            if not questions:
                raise HTTPException(status_code=400, detail="Could not generate MCQs from the PDF content")
                
            return {"status": "success", "questions": questions[:question_count]}
            
        else:  # QA pairs
            prompt_template = generate_qa_prompt(question_count)
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            runnable = prompt | llm
            
            qa_pairs = []
            for doc in documents[:3]:  # Limit to first 3 chunks
                try:
                    result = await runnable.ainvoke({"text": doc.page_content})
                    qas = parse_qas(result)
                    qa_pairs.extend(qa.dict() for qa in qas)
                    if len(qa_pairs) >= question_count:
                        break
                except Exception as e:
                    logger.warning(f"Error processing chunk: {str(e)}")
                    continue
                    
            if not qa_pairs:
                raise HTTPException(status_code=400, detail="Could not generate Q&A pairs from the PDF content")
                
            return {"status": "success", "questions": qa_pairs[:question_count]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_pdf(
    pdf_path: str = Form(...),
    content_type: str = Form(...),
    question_count: int = Form(DEFAULT_QUESTION_COUNT),
    difficulty: str = Form(DEFAULT_DIFFICULTY)
):
    try:
        if not pdf_path.startswith("/static/docs/"):
            raise HTTPException(status_code=400, detail="Invalid file path format")
        
        full_path = BASE_DIR / pdf_path.lstrip("/")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        output = await generate_output_files(
            str(full_path),
            content_type,
            question_count,
            difficulty
        )
        
        if "error" in output:
            raise HTTPException(status_code=400, detail=output["error"])
        
        return JSONResponse({
            "status": "success",
            "output_files": {
                "csv": output["csv_path"],
                "pdf": output["pdf_path"],
            },
            "content": output.get("mcqs", []) if content_type == "mcq" else output.get("qa_pairs", [])
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-paper")
async def generate_paper(request: GeneratePaperRequest):
    try:
        # Generate questions first if we have auto-generated questions
        questions = request.autoGeneratedQuestions or []
        mcq_questions = [q for q in questions if q.get("type") == "mcq"]
        desc_questions = [q for q in questions if q.get("type") == "qa"]

        # Generate HTML content
        html_parts = [
            '<div class="question-paper">',
            f'<h2>{request.institution}</h2>',
            f'<h3>{request.examType} - {request.year}</h3>',
            f'<p>Time: {request.time} | Marks: {request.marks}</p>',
            '<div class="instructions">',
            f'<p>{request.instructions.replace("\n", "<br>")}</p>',
            '</div>'
        ]

        question_number = 1
        for section in request.sections:
            html_parts.append(f'<div class="section">')
            html_parts.append(f'<h4>{section.name}</h4>')
            html_parts.append(f'<p>{section.instructions}</p>')

            # Add MCQs
            if section.mcqCount > 0:
                html_parts.append('<div class="mcqs">')
                for i in range(section.mcqCount):
                    if i < len(mcq_questions):
                        q = mcq_questions[i]
                        html_parts.append(f'<div class="question">')
                        html_parts.append(f'<p>{question_number}. {q["question"]}</p>')
                        html_parts.append('<div class="options">')
                        for opt in q["options"]:
                            html_parts.append(f'<p>{opt["letter"]}) {opt["text"]}</p>')
                        html_parts.append('</div>')
                        html_parts.append(f'<p>Correct: {q["correct_answer"]}</p>')
                        html_parts.append('</div>')
                    else:
                        html_parts.append(f'<p>{question_number}. [MCQ Placeholder]</p>')
                    question_number += 1
                html_parts.append('</div>')

            # Add descriptive questions
            if section.descCount > 0:
                html_parts.append('<div class="descriptive">')
                for i in range(section.descCount):
                    if i < len(desc_questions):
                        q = desc_questions[i]
                        html_parts.append(f'<div class="question">')
                        html_parts.append(f'<p>{question_number}. {q["question"]}</p>')
                        html_parts.append('</div>')
                    else:
                        html_parts.append(f'<p>{question_number}. [Descriptive Placeholder]</p>')
                    question_number += 1
                html_parts.append('</div>')

            html_parts.append('</div>')  # Close section

        html_parts.append('</div>')  # Close paper
        html_content = "\n".join(html_parts)

        return {
            "status": "success", 
            "html_content": html_content,
            "questions_used": len(mcq_questions) + len(desc_questions)
        }

    except Exception as e:
        logger.error(f"Paper generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
def generate_paper_html(request: GeneratePaperRequest) -> str:
    """Generate clean HTML content for the question paper"""
    # Header section
    html_parts = [
        f'<div class="question-paper">',
        f'<div class="paper-header">',
        f'<h2>{escape(request.institution)}</h2>',
        f'<h3>{escape(request.examType)} - {escape(request.year)}</h3>',
        f'<h4>{escape(request.course)}</h4>',
        f'<p>Time: {escape(request.time)} | Max Marks: {request.marks}</p>',
        '</div>',
        '<div class="exam-info">',
        '<div>Date: __________</div>',
        '<div>Roll No: __________</div>',
        '</div>'
    ]

    # Instructions with proper line breaks
    instructions = '<br>'.join(escape(line) for line in request.instructions.split('\n'))
    html_parts.extend([
        '<div class="instructions">',
        '<h5>General Instructions:</h5>',
        f'<p>{instructions}</p>',
        '</div>'
    ])

    # Process sections
    question_number = 1
    for section in request.sections:
        html_parts.extend([
            '<div class="section">',
            f'<h5 class="section-title">{escape(section.name)}</h5>',
            f'<p class="section-instructions">{escape(section.instructions)}</p>'
        ])

        # MCQs
        if section.mcqCount > 0:
            html_parts.append('<div class="mcq-questions">')
            for _ in range(section.mcqCount):
                html_parts.extend([
                    '<div class="question">',
                    f'<p>{question_number}. [MCQ Question Placeholder] ',
                    f'<span class="marks-info">[{section.marksPerQuestion} Mark(s)]</span></p>',
                    '</div>'
                ])
                question_number += 1
            html_parts.append('</div>')

        # Descriptive questions
        if section.descCount > 0:
            html_parts.append('<div class="desc-questions">')
            for _ in range(section.descCount):
                html_parts.extend([
                    '<div class="question">',
                    f'<p>{question_number}. [Descriptive Question Placeholder] ',
                    f'<span class="marks-info">[{section.marksPerQuestion} Mark(s)]</span></p>',
                    '</div>'
                ])
                question_number += 1
            html_parts.append('</div>')

        html_parts.append('</div>')  # Close section

    html_parts.append('</div>')  # Close question-paper
    return '\n'.join(html_parts)
@app.post("/generate-pdf")
async def generate_pdf_endpoint(request_data: dict):
    try:
        html_content = request_data.get("html_content")
        if not html_content:
            raise HTTPException(status_code=400, detail="Missing HTML content")
        
        timestamp = int(time.time())
        pdf_filename = f"paper_{timestamp}.pdf"
        pdf_path = output_dir / pdf_filename

        # Create PDF
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        story = []
        
        # Convert HTML to plain text paragraphs
        plain_text = re.sub(r'<[^>]+>', '', html_content)
        for line in plain_text.split('\n'):
            if line.strip():
                p = Paragraph(line, styles["Normal"])
                story.append(p)
                story.append(Spacer(1, 12))

        doc.build(story)
        return {"pdf_url": f"/static/output/{pdf_filename}"}

    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_pdf(html_content: str) -> str:
    """Generate PDF from cleaned HTML content"""
    timestamp = int(time.time())
    pdf_filename = f"paper_{timestamp}.pdf"
    pdf_path = output_dir / pdf_filename

    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # Define styles
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        spaceAfter=6
    )

    # Convert HTML to plain text for PDF
    plain_text = html_content
    # Remove HTML tags but keep line breaks
    plain_text = re.sub(r'<br\s*/?>', '\n', plain_text)
    plain_text = re.sub(r'<[^>]+>', '', plain_text)
    plain_text = re.sub(r'\n\s*\n', '\n\n', plain_text)  # Remove extra blank lines

    # Build story
    story = []
    for line in plain_text.split('\n'):
        if line.strip():
            p = Paragraph(escape(line), custom_style)
            story.append(p)
            story.append(Spacer(1, 12))

    doc.build(story)
    return f"/static/output/{pdf_filename}"

@app.get("/download/{filename:path}")
async def download_file(filename: str):
    try:
        file_path = output_dir / filename
        
        # Security check
        if not file_path.resolve().parent.samefile(output_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
            
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="File download failed")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
 