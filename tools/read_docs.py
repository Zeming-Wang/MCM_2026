import os
import sys
from pdfminer.high_level import extract_text
import docx

def read_pdf(path):
    try:
        text = extract_text(path)
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def read_docx(path):
    try:
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading DOCX: {e}"

if __name__ == "__main__":
    files = [
        r"d:\MCM_2026_O\assets\2026_MCM_Problem_C.pdf",
        r"d:\MCM_2026_O\assets\已有的论文和思路的集合.docx"
    ]
    
    for f in files:
        print(f"\n{'='*50}\nReading file: {f}\n{'='*50}")
        if not os.path.exists(f):
            print("File not found.")
            continue
            
        if f.lower().endswith('.pdf'):
            content = read_pdf(f)
        elif f.lower().endswith('.docx'):
            content = read_docx(f)
        else:
            content = "Unsupported format"
            
        # Print first 2000 chars to avoid overwhelming output, or full if needed.
        # User asked to read it, so I'll print enough to process.
        print(content[:5000]) 
