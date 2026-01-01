# -*- coding: utf-8 -*-
"""
嵌入向量生成模块

本模块提供文本嵌入向量生成功能，支持DeepSeek API和本地SentenceTransformer模型。
主要功能包括：
- 嵌入客户端初始化（自动选择API或本地模型）
- 文本分块处理
- 批量嵌入生成
- 错误处理和重试机制
"""

import logging
from typing import List, Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from coderag.config import (
    DEEPSEEK_API_BASE, 
    DEEPSEEK_API_KEY, 
    DEEPSEEK_EMBEDDING_MODEL
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 嵌入客户端类 - 支持DeepSeek API和本地模型
class EmbeddingClient:
    """嵌入客户端类，负责管理嵌入模型的初始化和使用
    
    属性:
        client: OpenAI客户端实例（用于DeepSeek API）
        model: 当前使用的模型名称
        local_model: 本地SentenceTransformer模型实例
        use_local: 是否使用本地模型
    """
    
    def __init__(self):
        """初始化嵌入客户端
        
        初始化策略:
        1. 优先尝试初始化DeepSeek API客户端
        2. 如果API配置不完整或初始化失败，则回退到本地模型
        3. 初始化本地SentenceTransformer模型
        """
        self.client = None  # OpenAI客户端实例
        self.model = None   # 当前使用的模型名称
        self.local_model = None  # 本地模型实例
        self.use_local = False   # 是否使用本地模型
        
        # 第一步：尝试初始化DeepSeek API客户端
        try:
            if DEEPSEEK_API_KEY and DEEPSEEK_EMBEDDING_MODEL and DEEPSEEK_API_BASE:
                # 配置完整，创建OpenAI客户端（适配DeepSeek API）
                self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
                self.model = DEEPSEEK_EMBEDDING_MODEL
                logger.info(f"DeepSeek客户端初始化成功，模型: {self.model} (用于嵌入生成)")
                logger.info(f"DeepSeek API基础URL: {DEEPSEEK_API_BASE}")
            else:
                # API配置不完整，使用本地模型
                logger.warning("DeepSeek配置不完整，回退到本地嵌入模型")
                self.use_local = True
        except Exception as e:
            # API初始化失败，使用本地模型
            logger.error(f"DeepSeek客户端初始化失败: {e}，回退到本地嵌入模型")
            self.use_local = True
        
        # 第二步：如果需要使用本地模型，则初始化本地模型
        if self.use_local:
            from coderag.config import EMBEDDING_DIM
            
            # 根据配置的嵌入维度选择模型
            if EMBEDDING_DIM == 768:
                model_options = [
                    'all-mpnet-base-v2',           # 768维模型
                    'multi-qa-mpnet-base-dot-v1',  # 768维备用模型
                    'all-MiniLM-L6-v2'             # 384维回退模型
                ]
            else:
                model_options = ['all-MiniLM-L6-v2']  # 384维模型
            
            # 尝试加载模型列表中的第一个可用模型
            for model_name in model_options:
                try:
                    self.local_model = SentenceTransformer(model_name)
                    logger.info(f"本地嵌入模型初始化成功: {model_name} ({EMBEDDING_DIM}维)")
                    break  # 成功加载后退出循环
                except Exception as e:
                    logger.error(f"模型 {model_name} 初始化失败: {e}")
                    continue  # 尝试下一个模型
            else:
                # 所有模型都加载失败
                logger.error("所有本地嵌入模型初始化失败")

# 创建全局嵌入客户端实例
embedding_client = EmbeddingClient()


def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 50) -> List[str]:
    """改进的文本分块函数，尊重句子边界
    
    这个函数将长文本分割成较小的块，同时保持语义完整性。
    分块策略：
    - 优先按句子分割
    - 保持句子完整性
    - 添加重叠区域以保持上下文连续性
    
    Args:
        text: 要分块的文本
        max_chars: 每个块的最大字符数（默认4000）
        overlap: 块之间的重叠字符数（默认50）
    
    Returns:
        List[str]: 文本块列表
    """
    import re
    
    # 清理文本并检查是否需要分块
    text = text.strip()
    if len(text) <= max_chars:
        return [text]  # 文本较短，无需分块
    
    # 按句子分割文本（基本但比字符分割更好）
    # 正则表达式：在句号、感叹号、问号后分割，但保持引号完整性
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []        # 存储最终的文本块
    current_chunk = []  # 当前正在构建的块
    current_length = 0  # 当前块的字符数
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # 检查添加当前句子是否会超过最大字符数
        if current_length + sentence_length > max_chars:
            if current_chunk:
                # 当前块已有内容，保存当前块并开始新块
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # 从当前块末尾计算重叠内容
                if len(chunk_text) > overlap:
                    overlap_content = chunk_text[-overlap:]  # 取末尾overlap个字符
                    current_chunk = [overlap_content, sentence]  # 新块包含重叠内容和当前句子
                    current_length = len(overlap_content) + sentence_length + 1  # +1用于空格
                else:
                    # 当前块太短，无法提供足够重叠
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                # 单个句子太长，按单词分割
                words = sentence.split()
                for i in range(0, len(words), max_chars):
                    # 按最大字符数分割长句子
                    chunk_text = ' '.join(words[i:i+max_chars])
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
        else:
            # 可以添加当前句子到当前块
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1用于句子间的空格
    
    # 处理最后一个块
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


@retry(
    stop=stop_after_attempt(3),      # 最多重试3次
    wait=wait_exponential(multiplier=0.5, max=8),  # 指数退避等待
    reraise=True,                    # 重试失败后重新抛出异常
)
def _embed_batch(inputs: List[str]) -> np.ndarray:
    """批量生成嵌入向量，使用DeepSeek API或本地模型
    
    这个函数是嵌入生成的核心，支持两种模式：
    1. 本地SentenceTransformer模型
    2. DeepSeek API调用
    
    Args:
        inputs: 输入文本列表
    
    Returns:
        np.ndarray: 形状为(n, d)的嵌入向量数组，n为输入数量，d为嵌入维度
    
    Raises:
        RuntimeError: 模型未初始化时抛出
    """
    # 检查是否使用本地模型
    if embedding_client.use_local:
        if embedding_client.local_model is None:
            raise RuntimeError("本地嵌入模型未初始化")
        
        # 记录本地模型使用信息
        logger.info(f"使用本地嵌入模型: {embedding_client.local_model}")
        logger.info(f"输入文本数量: {len(inputs)}")
        logger.info(f"第一个输入样本(截断): {inputs[0][:100]}...")
        
        # 使用本地模型生成嵌入向量
        embeddings = embedding_client.local_model.encode(
            inputs,
            convert_to_numpy=True,     # 转换为numpy数组
            normalize_embeddings=True, # 归一化嵌入向量
            show_progress_bar=False    # 不显示进度条
        )
        return np.array(embeddings, dtype="float32")
    
    # 回退到DeepSeek API
    if embedding_client.client is None:
        raise RuntimeError("DeepSeek嵌入客户端未初始化")
    
    # 记录API调用详细信息用于调试
    logger.info(f"调用DeepSeek嵌入API，模型: {embedding_client.model}")
    logger.info(f"DeepSeek API基础URL: {DEEPSEEK_API_BASE}")
    logger.info(f"输入文本数量: {len(inputs)}")
    logger.info(f"第一个输入样本(截断): {inputs[0][:100]}...")
    
    # 调用DeepSeek API生成嵌入向量
    response = embedding_client.client.embeddings.create(
        model=embedding_client.model,
        input=inputs,
        timeout=30,  # 30秒超时
    )
    
    # 从响应中提取嵌入向量并转换为numpy数组
    arr = np.array([d.embedding for d in response.data], dtype="float32")
    return arr


def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """生成文本的嵌入向量，使用DeepSeek API或本地模型
    
    这是主要的嵌入生成函数，处理整个流程：
    1. 文本预处理和验证
    2. 确保本地模型已初始化（作为备用）
    3. 文本分块处理
    4. 批量嵌入生成
    5. 错误处理和重试机制
    
    Args:
        text: 要生成嵌入向量的输入文本

    Returns:
        Optional[np.ndarray]: 形状为(1, d)的嵌入向量数组，失败时返回None
    """
    # 输入验证
    if not text or not text.strip():
        logger.warning("提供了空文本用于嵌入生成")
        return None

    # 确保本地模型已初始化（作为备用）
    if not embedding_client.local_model:
        from coderag.config import EMBEDDING_DIM
        
        # 根据嵌入维度选择模型
        if EMBEDDING_DIM == 768:
            model_options = [
                'all-mpnet-base-v2',           # 768维模型
                'multi-qa-mpnet-base-dot-v1',  # 768维备用模型
                'all-MiniLM-L6-v2'             # 384维回退模型
            ]
        else:
            model_options = ['all-MiniLM-L6-v2']  # 384维模型
        
        # 尝试初始化本地模型
        for model_name in model_options:
            try:
                embedding_client.local_model = SentenceTransformer(model_name)
                logger.info(f"本地嵌入模型初始化成功: {model_name} ({EMBEDDING_DIM}维)")
                break
            except Exception as e:
                logger.error(f"模型 {model_name} 初始化失败: {e}")
                continue
        else:
            # 所有模型初始化都失败
            logger.error("所有本地嵌入模型初始化失败")
            return None

    try:
        # 记录嵌入生成开始
        logger.info(f"开始生成嵌入向量，文本长度: {len(text)}")

        # 步骤1: 文本分块
        chunks = _chunk_text(text, max_chars=4000)
        
        # 步骤2: 批量生成嵌入向量
        vecs = _embed_batch(chunks)  # 形状为(n, d)

        # 步骤3: 对分块嵌入向量进行平均，得到稳定的单一向量
        avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
        logger.info(f"嵌入向量生成成功，形状: {avg.shape}")
        return avg

    except Exception as e:
        logger.error(f"嵌入向量生成失败: {e}")
        
        # 详细调试404错误（API端点不存在）
        if "404" in str(e) and not embedding_client.use_local:
            logger.error("DeepSeek API返回404错误，切换到本地嵌入模型...")
            logger.error(f"1. API基础URL: {DEEPSEEK_API_BASE}")
            logger.error(f"2. 模型名称: {DEEPSEEK_EMBEDDING_MODEL}")
            logger.error(f"3. API密钥: {'有效格式' if DEEPSEEK_API_KEY and len(DEEPSEEK_API_KEY) > 10 else '无效或缺失'}")
            logger.error(f"4. 网络连接到 {DEEPSEEK_API_BASE}")
            
            # 切换到本地模型用于后续请求
            embedding_client.use_local = True
            
            # 使用本地模型重试
            try:
                logger.info("使用本地嵌入模型重试...")
                chunks = _chunk_text(text, max_chars=4000)
                vecs = _embed_batch(chunks)
                avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
                logger.info(f"使用本地模型成功生成嵌入向量，形状: {avg.shape}")
                return avg
            except Exception as local_e:
                logger.error(f"本地嵌入生成也失败: {local_e}")
                return None
        
        # 对于其他错误，如果尚未使用本地模型，则直接尝试本地模型
        if not embedding_client.use_local:
            logger.error("DeepSeek API失败，切换到本地嵌入模型...")
            embedding_client.use_local = True
            
            # 使用本地模型重试
            try:
                logger.info("使用本地嵌入模型重试...")
                chunks = _chunk_text(text, max_chars=4000)
                vecs = _embed_batch(chunks)
                avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
                logger.info(f"使用本地模型成功生成嵌入向量，形状: {avg.shape}")
                return avg
            except Exception as local_e:
                logger.error(f"本地嵌入生成也失败: {local_e}")
                return None
        
        # 所有重试都失败，返回None
        return None
