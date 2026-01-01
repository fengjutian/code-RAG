#!/usr/bin/env python3
"""
Test script to verify that SentenceTransformer models can be loaded correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer

def test_model_loading():
    """Test loading different embedding models."""
    
    # Test models that should work
    test_models = [
        'all-mpnet-base-v2',  # 768 dimensions
        'multi-qa-mpnet-base-dot-v1',  # 768 dimensions
        'paraphrase-multilingual-mpnet-base-v2',  # 768 dimensions
        'all-MiniLM-L6-v2',  # 384 dimensions
    ]
    
    for model_name in test_models:
        try:
            print(f"\n=== Testing model: {model_name} ===")
            
            # Load the model
            model = SentenceTransformer(model_name)
            
            # Test encoding
            test_text = "This is a test sentence for embedding generation."
            embedding = model.encode(test_text)
            
            print(f"✓ Model loaded successfully")
            print(f"  Dimensions: {embedding.shape}")
            print(f"  Model info: {model}")
            
        except Exception as e:
            print(f"✗ Failed to load model {model_name}: {e}")
            print(f"  Error type: {type(e).__name__}")

def main():
    print("Testing SentenceTransformer model loading...")
    print("=" * 60)
    
    test_model_loading()
    
    print("\n" + "=" * 60)
    print("Test completed.")

if __name__ == "__main__":
    main()