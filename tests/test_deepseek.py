import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get DeepSeek configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBEDDING_MODEL")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")

logger.info("Testing DeepSeek API Connection...")
logger.info(f"API Key: {DEEPSEEK_API_KEY[:10]}..." if DEEPSEEK_API_KEY else "API Key: Missing")
logger.info(f"Embedding Model: {DEEPSEEK_EMBEDDING_MODEL}")
logger.info(f"API Base: {DEEPSEEK_API_BASE}")

if not all([DEEPSEEK_API_KEY, DEEPSEEK_EMBEDDING_MODEL, DEEPSEEK_API_BASE]):
    logger.error("Missing required configuration. Please check .env file.")
    sys.exit(1)

try:
    # Test with v1 suffix
    logger.info("\nTesting with /v1 suffix...")
    client_v1 = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=f"{DEEPSEEK_API_BASE}/v1")
    
    # Test embedding API with different model names
    embedding_succeeded = False
    for model_name in [DEEPSEEK_EMBEDDING_MODEL, "text-embedding-ada-002"]:
        try:
            logger.info(f"Testing with model: {model_name}")
            response = client_v1.embeddings.create(
                model=model_name,
                input=["test"]
            )
            logger.info(f"✅ Embedding API test succeeded with model {model_name}!")
            logger.info(f"Embedding dimension: {len(response.data[0].embedding)}")
            embedding_succeeded = True
            break
        except Exception as e:
            logger.error(f"❌ Embedding API test failed with model {model_name}: {e}")
    if not embedding_succeeded:
        logger.error("All embedding API tests failed with /v1 suffix")
    
except Exception as e:
    logger.error(f"❌ Embedding API test failed with /v1 suffix: {e}")

# Test without v1 suffix
logger.info("\nTesting without /v1 suffix...")
try:
    client_no_v1 = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
    
    # Test embedding API
    try:
        response = client_no_v1.embeddings.create(
            model=DEEPSEEK_EMBEDDING_MODEL,
            input=["test"]
        )
        logger.info("✅ Embedding API test succeeded!")
        logger.info(f"Embedding dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        logger.error(f"❌ Embedding API test failed without /v1 suffix: {e}")
except Exception as e:
    logger.error(f"❌ Test without /v1 suffix failed: {e}")

# Test local embedding model
try:
    logger.info("\nTesting local embedding model...")
    from sentence_transformers import SentenceTransformer
    
    # Initialize local model
    local_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    logger.info("✅ Local model initialized successfully")
    
    # Test embedding generation
    embeddings = local_model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True)
    logger.info("✅ Local embedding generation succeeded!")
    logger.info(f"Embedding dimension: {len(embeddings[0])}")
    logger.info(f"First few embedding values: {embeddings[0][:5]}")
    
except Exception as e:
    logger.error(f"❌ Local embedding model test failed: {e}")

try:
    # Test chat API
    logger.info("\nTesting chat API...")
    DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL")
    if DEEPSEEK_CHAT_MODEL:
        client_chat = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=f"{DEEPSEEK_API_BASE}/v1")
        response = client_chat.chat.completions.create(
            model=DEEPSEEK_CHAT_MODEL,
            messages=[{"role": "user", "content": "Hello"}]
        )
        logger.info("✅ Chat API test succeeded!")
        logger.info(f"Response: {response.choices[0].message.content}")
    else:
        logger.warning("Chat model not configured, skipping chat API test.")
        
except Exception as e:
    logger.error(f"❌ Chat API test failed: {e}")

logger.info("\nTesting completed.")