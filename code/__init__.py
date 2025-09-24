"""Project package for ML pipeline modules.

This file ensures the repository-level ``code`` directory is treated as a
package so imports like ``code.models.train_model`` resolve to the project
modules instead of the Python stdlib ``code`` module.
"""

__all__ = [
    "deployment",
    "models",
]
