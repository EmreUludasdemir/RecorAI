# Contribution Guidelines

Thank you for your interest in contributing to RecorAI! 🎉

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/RecorAI/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📋 Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

### Example

```python
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) score.

    Args:
        pred: Predicted mask tensor
        target: Ground truth mask tensor
        threshold: Binarization threshold

    Returns:
        IoU score as float
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-7) / (union + 1e-7)).item()
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models.py -v
```

## 📝 Commit Messages

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## 🏷️ Labels

| Label              | Description                |
| ------------------ | -------------------------- |
| `bug`              | Something isn't working    |
| `enhancement`      | New feature request        |
| `documentation`    | Documentation improvements |
| `good first issue` | Good for newcomers         |
| `help wanted`      | Extra attention needed     |

## 📜 Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## ⚠️ Competition Period

During the active competition period (until January 8, 2026), please be mindful of:

- Not sharing specific competition strategies publicly
- Respecting Kaggle's competition rules
- Focusing contributions on general improvements

After the competition ends, all contributions are welcome!

---

Thank you for contributing! 🙌
