"""
Test configuration for SV detection.
"""

import pytest
import tempfile
from pathlib import Path

# Test configuration constants
TEST_DATA_DIR = "test_data"
TEST_OUTPUT_DIR = "test_output"

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
