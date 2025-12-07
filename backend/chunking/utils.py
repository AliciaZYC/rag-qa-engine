"""
Utility functions for text chunking
"""
import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

# Download punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass  # punkt_tab may not be needed in older NLTK versions

def token_count(text: str) -> int:
    """
    Estimate token count (approximation: 1 token ≈ 4 characters)
    For more accurate counting, use transformers tokenizer
    """
    return len(text) // 4

def split_by_sentences(text: str, max_tokens: int = 400, overlap: int = 1) -> List[str]:
    """
    Split text into chunks at sentence boundaries with token limit
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = token_count(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
                current_tokens = sum(token_count(s) for s in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def detect_section_header(line: str) -> bool:
    """
    Detect if a line is a section header in legal documents
    
    Patterns:
    - "Article 5"
    - "Section 3.2"
    - "3.2 Indemnification"
    - "WHEREAS"
    - "NOW, THEREFORE"
    """
    line = line.strip()
    
    if not line:
        return False
    
    # All caps short lines (common in legal docs)
    if line.isupper() and len(line.split()) <= 5:
        return True
    
    # Numbered sections
    patterns = [
        r'^(Article|Section|Clause|Provision)\s+\d+',
        r'^\d+\.\d+\s+',  # 3.2 Title
        r'^[IVXLCDM]+\.',  # Roman numerals
        r'^(WHEREAS|NOW, THEREFORE|WITNESSETH)',
    ]
    
    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    return False

def extract_provision_label(metadata: dict) -> str:
    """
    Extract provision label from LEDGAR metadata
    
    LEDGAR has columns like: 'indemnification', 'governing_law', 'confidentiality'
    Returns the first label found
    """
    # Common LEDGAR provision types
    provision_types = [
        'governing_law', 'indemnification', 'confidentiality', 
        'termination', 'assignment', 'amendment', 'waiver',
        'severability', 'entire_agreement', 'notice',
        'force_majeure', 'dispute_resolution', 'liability'
    ]
    
    for provision in provision_types:
        if provision in metadata and metadata[provision]:
            return provision
    
    return 'general'

def normalize_legal_text(text: str) -> str:
    """
    Normalize legal text while preserving legal symbols
    
    - Normalize whitespace
    - Preserve legal symbols (§, ¶, ©, ®, ™)
    - Preserve section numbering
    """
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text