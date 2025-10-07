import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "mps: mark test as requiring Apple MPS")
    config.addinivalue_line("markers", "mlx: mark test as requiring MLX backend")
