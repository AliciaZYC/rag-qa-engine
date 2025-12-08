"""
Pre-download embedding models and NLTK data
Run: python download_models.py
"""
from sentence_transformers import SentenceTransformer
import nltk
import sys

def download_nltk_data():
    """Download required NLTK data"""
    print("\n" + "="*60)
    print("Downloading NLTK Data")
    print("="*60)
    
    try:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        return False

def download_model(model_name, description):
    """Download and test a model"""
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        print("Downloading... (this may take a few minutes)")
        model = SentenceTransformer(model_name)
        
        # Test encoding
        test_text = "This is a test sentence for the legal document system."
        embedding = model.encode([test_text])
        
        print(f"  Model downloaded successfully")
        print(f"  Embedding dimension: {embedding.shape[1]}")
        print(f"  Model stored in: ~/.cache/torch/sentence_transformers/")
        
        return True
    except Exception as e:
        print(f"  Failed to download {model_name}")
        print(f"  Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("DOWNLOADING REQUIRED DATA & MODELS")
    print("="*60)
    print("\nThis script will download:")
    print("  1. NLTK punkt tokenizer (~1MB)")
    print("  2. all-MiniLM-L6-v2 model (~80MB)")
    print("  3. legal-bert-base-uncased model (~420MB)")
    print("\nTotal size: ~500MB")
    
    results = []
    
    # Step 1: Download NLTK data
    results.append(download_nltk_data())
    
    # Step 2: Download MiniLM (fallback model)
    results.append(
        download_model(
            'all-MiniLM-L6-v2',
            'MiniLM - Fast fallback model (384 dimensions)'
        )
    )
    
    # Step 3: Download LegalBERT (primary model)
    results.append(
        download_model(
            'nlpaueb/legal-bert-base-uncased',
            'LegalBERT - Legal domain optimized (768 dimensions)'
        )
    )
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded: {sum(results)}/3 items")
    
    if all(results):
        print("\nAll data and models ready for use!")
        print("\nYou can now run:")
        print("  docker-compose exec backend python db/ingest_data.py")
        return 0
    else:
        print("\nSome items failed to download.")
        print("The system will still work with available models.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)