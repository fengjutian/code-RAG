import sys
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 重启嵌入客户端
try:
    # 直接测试新模型，不使用EmbeddingClient类的API回退机制
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info(f"模型加载成功: all-MiniLM-L6-v2")
    logger.info(f"模型维度: {model.get_sentence_embedding_dimension()}")
    
    logger.info("测试新的嵌入模型 all-MiniLM-L6-v2")
    
    # 测试模型相似度
    test_sentences = [
        ("What is pip?", "pip is a package installer for Python", "相关"),
        ("What is pip?", "How to install a Python package with pip", "相关"),
        ("What is pip?", "Python virtual environments", "不相关"),
        ("What is pip?", "How to bake a cake", "不相关")
    ]
    
    for sent1, sent2, label in test_sentences:
        emb1 = model.encode(sent1)
        emb2 = model.encode(sent2)
        similarity = model.similarity(emb1, emb2).item()
        logger.info(f"{label}句子: '{sent1}' 和 '{sent2}'")
        logger.info(f"相似度: {similarity:.4f}")
        logger.info("---")
        
    logger.info("测试完成")
    
except Exception as e:
    logger.error(f"测试失败: {e}")
    import traceback
    traceback.print_exc()