# 嵌入词向量查询效果不佳的原因分析与解决方案

## 当前系统的查询效果分析

通过测试脚本的运行结果，我们可以看到以下几个主要问题：

### 1. 相似度得分普遍较低
所有查询的相似度得分都在 **0.4-0.57之间**，这个范围的相似度表明向量表示的质量不高，没有很好地捕捉到文本的语义相似性。理想情况下，相关文档的相似度应该在0.7以上。

### 2. 检索结果相关性差
例如：
- 查询 `"What is numpy?"` 时，前几个结果是 sympy 和 scipy 相关文件，而不是 numpy 核心文件
- 查询 `"How to install a Python package"` 时，返回的是一些不相关的文件

### 3. 文本覆盖不全面
当前索引中包含了14974个向量，但从查询结果来看，很多关键文件没有被正确检索到。

## 导致查询效果不好的根本原因

### 1. 文本分块策略过于简单
当前使用的是简单的字符分块（`_chunk_text`函数），将文本按照固定长度（4000字符）分割，这种方法没有考虑语义完整性：

```python
def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Naive chunking by characters to avoid overly long inputs."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
```

这种方法会导致：
- 句子被截断，破坏语义完整性
- 上下文信息丢失
- 重要概念被分割到不同块中

### 2. 嵌入模型的选择和配置问题
当前使用的是轻量级的本地模型 `paraphrase-MiniLM-L6-v2`：

- 这个模型是为通用文本设计的，可能不适合技术文档（尤其是代码）
- 模型维度只有384维，表达能力有限
- 没有针对代码文档进行专门的微调

### 3. 索引策略过于简单
使用的是简单的 FAISS `IndexFlatIP`（内积索引）：

- 没有进行任何降维处理（如 PCA）
- 没有使用更高效的索引结构（如 IVF，HNSW）
- 没有进行参数调优

### 4. 文本预处理不足
当前系统没有对输入文本进行适当的预处理：

- 没有去除代码中的注释和空白
- 没有对不同类型的文档（如 Python 文件、Markdown）进行针对性处理
- 没有提取关键信息（如函数名、类名）作为索引依据

### 5. 文档嵌入的计算方式
当前系统对整个文档进行分块后，将所有块的嵌入向量进行平均：

```python
def generate_embeddings(text: str) -> Optional[np.ndarray]:
    # ...
    chunks = _chunk_text(text, max_chars=4000)
    vecs = _embed_batch(chunks)  # shape (n, d)
    # Average chunk embeddings for a stable single vector
    avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
    # ...
```

这种平均方式会导致：
- 重要信息被稀释
- 文档的独特特征丢失
- 查询和文档之间的匹配度降低

## 解决方案

### 1. 改进文本分块策略

将简单的字符分块改为基于语义的分块：

```python
def _chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Improved chunking that respects sentence boundaries."""
    import re
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed max_tokens
        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                # Add overlap from end of current chunk to next chunk
                overlap_content = ' '.join(current_chunk[-overlap:]) if len(current_chunk) > overlap else ' '.join(current_chunk)
                current_chunk = [overlap_content, sentence]
                current_length = len(overlap_content) + sentence_length + 1
            else:
                # Single sentence too long, split by words
                words = sentence.split()
                for i in range(0, len(words), max_tokens):
                    chunks.append(' '.join(words[i:i+max_tokens]))
                current_chunk = []
                current_length = 0
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### 2. 优化嵌入模型

- **升级到更强大的模型**：使用 `all-mpnet-base-v2` 或 `all-MiniLM-L12-v2` 等更强大的模型
- **考虑使用代码专用模型**：如 `sentence-transformers/paraphrase-codebert-small-v1`

```python
# 初始化更强大的模型
self.local_model = SentenceTransformer('all-mpnet-base-v2')
```

### 3. 改进索引策略

- **使用降维技术**：如 PCA 降维到 128 或 256 维
- **使用更高效的索引结构**：如 IVF（倒排文件索引）或 HNSW（分层可导航小世界图）

```python
# 使用 IVF 索引
nlist = 100  # 聚类中心数量
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)  # 训练索引
index.add(embeddings)  # 添加向量
```

### 4. 增强文本预处理

- **针对代码文件的预处理**：
  - 去除注释和空白行
  - 提取函数名、类名和变量名作为元数据
  - 保留代码结构信息

- **针对文档文件的预处理**：
  - 去除格式标记（如 Markdown 语法）
  - 提取标题、章节等结构信息

### 5. 改进文档嵌入的计算方式

- **使用加权平均**：根据块的重要性（如是否包含标题、关键字）进行加权
- **使用 CLS 标记**：对于支持 CLS 标记的模型，使用 CLS 标记的输出作为文档表示
- **使用最大池化**：保留块嵌入中的最大特征值

```python
def generate_embeddings(text: str) -> Optional[np.ndarray]:
    # ...
    chunks = _chunk_text(text, max_chars=4000)
    vecs = _embed_batch(chunks)  # shape (n, d)
    
    # 使用最大池化代替平均池化
    max_embedding = np.max(vecs, axis=0, dtype=np.float32).reshape(1, -1)
    return max_embedding
```

### 6. 实现分层索引

- **文档级索引**：保留完整文档的嵌入
- **块级索引**：同时索引文档的各个块
- **元数据索引**：索引文档的元数据（如文件名、目录结构、标签等）

### 7. 查询扩展和重排序

- **查询扩展**：在查询中添加相关术语
- **重排序**：使用机器学习模型对初始检索结果进行重排序

## 实施建议

### 短期改进（1-2天）

1. **改进文本分块策略**：实现基于句子的分块
2. **升级嵌入模型**：使用 `all-mpnet-base-v2`
3. **优化索引结构**：使用 IVF 索引

### 中期改进（1周）

1. **增强文本预处理**：针对不同类型的文档实现专门的预处理
2. **改进文档嵌入计算**：使用加权平均或最大池化
3. **实现查询扩展**：添加同义词和相关术语

### 长期改进（1个月）

1. **实现分层索引**：文档级、块级和元数据索引
2. **集成重排序机制**：使用机器学习模型提升检索结果质量
3. **针对技术文档的微调**：使用代码文档数据集对模型进行微调

## 预期效果

通过以上改进，我们预期：

1. 相似度得分提升到 **0.7-0.95之间**
2. 检索结果的相关性显著提高
3. 查询响应时间保持在可接受范围内
4. 系统能够更好地处理各种类型的技术文档
