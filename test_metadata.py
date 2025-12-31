import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 导入所需模块
    from coderag.index import load_index, get_metadata, add_to_index, clear_index
    from coderag.embeddings import generate_embeddings
    
    logger.info("开始测试元数据存储和检索逻辑...")
    
    # 1. 清理并重新初始化索引
    logger.info("1. 清理并重新初始化索引...")
    clear_index()
    index = load_index()
    metadata_list = get_metadata()
    logger.info(f"✅ 索引初始化成功，包含 {index.ntotal} 个向量")
    
    # 2. 添加测试数据
    test_data = [
        {"content": "pip is a package installer for Python", "filename": "pip_intro.py", "filepath": "test/pip_intro.py"},
        {"content": "How to install packages using pip install command", "filename": "pip_install.py", "filepath": "test/pip_install.py"},
        {"content": "Python virtual environments create isolated development environments", "filename": "venv.py", "filepath": "test/venv.py"},
        {"content": "numpy is a library for numerical computing in Python", "filename": "numpy_intro.py", "filepath": "test/numpy_intro.py"}
    ]
    
    logger.info("2. 添加测试数据到索引...")
    for item in test_data:
        embedding = generate_embeddings(item["content"])
        if embedding is not None:
            add_to_index(embedding, item["content"], item["filename"], item["filepath"])
        else:
            logger.error(f"❌ 无法为 {item['filename']} 生成嵌入")
    
    # 保存并重新加载索引
    from coderag.index import save_index
    save_index()
    index = load_index()
    metadata_list = get_metadata()
    logger.info(f"✅ 测试数据添加完成，当前索引包含 {index.ntotal} 个向量")
    
    # 3. 测试元数据检索
    logger.info("3. 测试元数据检索...")
    for i in range(index.ntotal):
        if i < len(metadata_list):
            logger.info(f"  向量 {i}: 文件={metadata_list[i]['filepath']}, 内容预览={metadata_list[i]['content'][:50]}...")
        else:
            logger.error(f"  向量 {i} 没有对应的元数据!")
    
    # 4. 测试查询检索是否能正确匹配元数据
    logger.info("4. 测试查询检索是否能正确匹配元数据...")
    test_query = "What is pip?"
    embedding = generate_embeddings(test_query)
    k = 3
    distances, indices = index.search(embedding, k)
    
    logger.info(f"   查询: {test_query}")
    logger.info(f"   检索结果:")
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(metadata_list):
            logger.info(f"     {i+1}. 相似度: {1 - distances[0][i]:.4f}, 文件: {metadata_list[idx]['filepath']}")
            logger.info(f"        内容预览: {metadata_list[idx]['content'][:100]}...")
        else:
            logger.error(f"     {i+1}. 无效的向量索引: {idx}")
    
    logger.info("🎉 元数据存储和检索逻辑测试完成!")
    
except Exception as e:
    logger.error(f"❌ 测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)