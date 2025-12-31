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
    from coderag.index import load_index, get_metadata
    from coderag.embeddings import generate_embeddings
    
    logger.info("开始测试嵌入查询效果...")
    
    # 1. 测试FAISS索引加载
    logger.info("1. 加载FAISS索引...")
    index = load_index()
    metadata_list = get_metadata()
    if index:
        logger.info(f"✅ FAISS索引加载成功，包含 {index.ntotal} 个向量")
    else:
        logger.error("❌ FAISS索引加载失败")
        sys.exit(1)
    
    # 2. 测试几个不同的查询
    test_queries = [
        "What is pip?",
        "How to install a Python package",
        "Python virtual environment",
        "What is numpy?"
    ]
    
    for query in test_queries:
        logger.info(f"\n2. 查询: {query}")
        
        # 生成查询嵌入
        embedding = generate_embeddings(query)
        if embedding is None:
            logger.error("❌ 嵌入生成失败")
            continue
        
        # 进行向量检索
        k = 5  # 检索前5个结果
        distances, indices = index.search(embedding, k)
        
        relevant_files = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(metadata_list):  # 确保索引有效
                relevant_files.append({
                    "file": metadata_list[idx]["filepath"],
                    "distance": distances[0][i],
                    "similarity": 1 - distances[0][i]  # 转换为相似度
                })
        
        if relevant_files:
            logger.info(f"✅ 检索到 {len(relevant_files)} 个相关文件:")
            for i, file_info in enumerate(relevant_files):
                logger.info(f"   {i+1}. {file_info['file']} (相似度: {file_info['similarity']:.4f})")
        else:
            logger.warning("⚠️ 没有检索到相关文件")
    
    logger.info("\n🎉 查询测试完成!")
    
except Exception as e:
    logger.error(f"❌ 测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
