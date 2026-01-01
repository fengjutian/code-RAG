#!/usr/bin/env python3
"""Test script to verify which embedding model is being used and its dimension."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coderag.embeddings import embedding_client
from sentence_transformers import SentenceTransformer

print("Testing Embedding Model Configuration...")
print("=" * 50)

# Test the current embedding client
print("1. Current Embedding Client Status:")
print(f"   Use Local: {embedding_client.use_local}")
print(f"   Model: {embedding_client.model}")
print(f"   Local Model: {embedding_client.local_model}")

# Test the local model directly
print("\n2. Testing Local Model:")
try:
    # Test with the model we want to use
    test_model = SentenceTransformer('all-mpnet-base-v2')
    test_text = "This is a test sentence."
    embedding = test_model.encode([test_text])
    print(f"   ✅ all-mpnet-base-v2 model works!")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dimension: {embedding.shape[1]}")
except Exception as e:
    print(f"   ❌ all-mpnet-base-v2 failed: {e}")

# Test the old model for comparison
print("\n3. Testing Old Model (for comparison):")
try:
    old_model = SentenceTransformer('all-MiniLM-L6-v2')
    test_text = "This is a test sentence."
    embedding = old_model.encode([test_text])
    print(f"   ✅ all-MiniLM-L6-v2 model works!")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dimension: {embedding.shape[1]}")
except Exception as e:
    print(f"   ❌ all-MiniLM-L6-v2 failed: {e}")

print("\n4. Testing generate_embeddings function:")
from coderag.embeddings import generate_embeddings

test_text = "Test sentence for embedding generation."
result = generate_embeddings(test_text)

if result is not None:
    print(f"   ✅ generate_embeddings works!")
    print(f"   Embedding shape: {result.shape}")
    print(f"   Embedding dimension: {result.shape[1]}")
else:
    print("   ❌ generate_embeddings returned None")

print("\n" + "=" * 50)
print("If the dimension is still 384, check:")
print("1. Python process restart (close and reopen terminal)")
print("2. Code changes are saved and loaded")
print("3. No cached modules")