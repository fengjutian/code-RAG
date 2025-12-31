# 🛠️ Development Guide

## Setting Up Development Environment

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/CodeRAG.git
cd CodeRAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

> The requirements file delegates to `-e .[dev]`, so you can also run
> `pip install -e .[dev]` directly if you prefer editable installs.
```

### 2. Configure Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This will run code quality checks on every commit:
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting and style checks
- **MyPy**: Type checking
- **Basic hooks**: Trailing whitespace, file endings, etc.

### 3. Environment Variables

Copy `example.env` to `.env` and configure:

```bash
cp example.env .env
```

Required variables:
```env
OPENAI_API_KEY=your_key_here  # Required for embeddings and chat
WATCHED_DIR=/path/to/code     # Directory to index (default: current dir)
```

## Code Quality Standards

### Type Hints
All functions should have type hints:

```python
def process_file(filepath: str, content: str) -> Optional[np.ndarray]:
    \"\"\"Process a file and return embeddings.\"\"\"
    ...
```

### Error Handling
Use structured logging and proper exception handling:

```python
import logging
logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {str(e)}")
    return None
```

### Documentation
Use concise docstrings for public functions:

```python
def search_code(query: str, k: int = 5) -> List[Dict[str, Any]]:
    \"\"\"Search the FAISS index using a text query.

    Args:
        query: The search query text
        k: Number of results to return

    Returns:
        List of search results with metadata
    \"\"\"
```

## Testing Your Changes

### Manual Testing
```bash
# Test backend indexing
python main.py

# Test Streamlit UI (separate terminal)
streamlit run app.py
```

### Code Quality Checks
```bash
pre-commit run --all-files
```

If you need to run a specific tool locally:

```bash
black .
isort .
flake8 .
mypy .
```

## Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Add logging**: Use the logger for all operations
3. **Add type hints**: Follow existing patterns
4. **Handle errors**: Graceful degradation and user-friendly messages
5. **Update tests**: Add tests for new functionality
6. **Update docs**: Update README if needed

## Architecture Guidelines

### Keep It Simple
- Maintain the single-responsibility principle
- Avoid unnecessary abstractions
- Focus on the core RAG functionality

### Error Handling Strategy
- Log errors with context
- Return None/empty lists for failures
- Show user-friendly messages in UI
- Don't crash the application

### Performance Considerations
- Limit search results (default: 5)
- Truncate long content for context
- Cache embeddings when possible
- Monitor memory usage with large codebases

## Debugging Tips

### Enable Debug Logging
```python
logging.basicConfig(level=logging.DEBUG)
```

### Check Index Status
```python
from coderag.index import inspect_metadata
inspect_metadata(5)  # Show first 5 entries
```

### Test Embeddings
```python
from coderag.embeddings import generate_embeddings
result = generate_embeddings("test code")
print(f"Shape: {result.shape if result is not None else 'None'}")
```

## Common Development Issues

**Import Errors**
- Ensure you're in the virtual environment
- Check PYTHONPATH includes project root
- Verify all dependencies are installed

**OpenAI API Issues**
- Check API key validity
- Monitor rate limits and usage
- Test with a simple embedding request

**FAISS Index Corruption**
- Delete existing index files and rebuild
- Check file permissions
- Ensure consistent embedding dimensions

## Routine Maintenance

- **Regenerate the FAISS index** after large code refactors: `python scripts/initialize_index.py`.
- **Rotate environment secrets** by updating `.env` or your deployment variables, then restarting services.
- **Refresh dependencies** with `pip install --upgrade -r requirements.txt` and run `pre-commit run --all-files` plus `pytest -q`.
- **Keep hooks current** using `pre-commit autoupdate` followed by a commit once checks pass.

## Project Structure

```
CodeRAG/
├── coderag/              # Core library
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── embeddings.py     # OpenAI integration
│   ├── index.py          # FAISS operations
│   ├── search.py         # Search functionality
│   └── monitor.py        # File monitoring
├── scripts/              # Utility scripts
├── tests/                # Test files
├── .github/              # GitHub workflows
├── main.py              # Backend service
├── app.py               # Streamlit frontend
├── prompt_flow.py       # RAG orchestration
└── requirements.txt     # Dependencies
```
