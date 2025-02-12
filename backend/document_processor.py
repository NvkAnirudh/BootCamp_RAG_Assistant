import os
from typing import List, Dict, Any
from pathlib import Path
import pypdf
import nltk
from nltk.tokenize import sent_tokenize

class DocumentProcessor:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + ' '
        return text.strip()
    
    def create_contextual_chunks(self, text: str, chunk_size: int = 128) -> List[Dict[str, str]]:
        """Create chunks with surrounding context."""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            # Create main chunk
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = ' '.join(chunk_sentences)
            
            # Create context (previous and next sentences)
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(sentences), i + chunk_size + self.window_size)
            context_sentences = sentences[start_idx:i] + sentences[i + chunk_size:end_idx]
            context = ' '.join(context_sentences)
            
            chunks.append({
                'content': chunk_text,
                'context': context
            })
        
        return chunks

    def process_documents(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """Process all PDFs with context."""
        all_documents = []
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        if len(pdf_files) == 0:
            raise ValueError(f"No PDF files found in directory: {pdf_dir}")
            
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            print(f"Extracted {len(text)} characters from {pdf_file.name}")
            
            if not text.strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue
                
            chunks = self.create_contextual_chunks(text)
            print(f"Created {len(chunks)} chunks from {pdf_file.name}")
            
            for i, chunk in enumerate(chunks):
                all_documents.append({
                    'source': pdf_file.name,
                    'chunk_id': f"{pdf_file.stem}_chunk_{i}",
                    'content': chunk['content'],
                    'context': chunk['context']
                })
        
        print(f"Total documents processed: {len(all_documents)}")
        return all_documents