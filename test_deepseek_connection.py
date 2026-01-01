#!/usr/bin/env python3
"""Test script to verify DeepSeek API connection for embeddings."""

import os
import sys
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get DeepSeek configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBEDDING_MODEL")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")

print("Testing DeepSeek API Connection...")
print(f"API Key: {DEEPSEEK_API_KEY[:10]}..." if DEEPSEEK_API_KEY else "API Key: Missing")
print(f"Embedding Model: {DEEPSEEK_EMBEDDING_MODEL}")
print(f"API Base: {DEEPSEEK_API_BASE}")

if not all([DEEPSEEK_API_KEY, DEEPSEEK_EMBEDDING_MODEL, DEEPSEEK_API_BASE]):
    print("❌ Missing required configuration. Please check .env file.")
    sys.exit(1)

# Test with different API base URL formats
test_urls = [
    DEEPSEEK_API_BASE,
    f"{DEEPSEEK_API_BASE}/v1",
]

for test_url in test_urls:
    print(f"\nTesting with URL: {test_url}")
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=test_url)
        
        # Test embedding API
        response = client.embeddings.create(
            model=DEEPSEEK_EMBEDDING_MODEL,
            input=["Hello, this is a test sentence."],
            timeout=30,
        )
        
        embeddings = response.data[0].embedding
        print(f"✅ Success! Embedding dimension: {len(embeddings)}")
        print(f"First 5 values: {embeddings[:5]}")
        break
        
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "="*50)
print("If all tests failed, check:")
print("1. API key validity")
print("2. API base URL format")
print("3. Network connectivity")
print("4. Model name compatibility")