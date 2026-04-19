"""
Tests for security validation utilities.
"""

import pytest

from torchregress.utils.security import validate_url


def test_validate_url_valid() -> None:
    """Test that valid URLs are accepted."""
    assert validate_url("http://example.com") == "http://example.com"
    assert validate_url("https://example.com/path/to/file") == "https://example.com/path/to/file"
    assert (
        validate_url("https://zenodo.org/api/records/12345")
        == "https://zenodo.org/api/records/12345"
    )


def test_validate_url_invalid_scheme() -> None:
    """Test that URLs with invalid schemes raise ValueError."""
    with pytest.raises(ValueError, match="URL scheme 'file' is not allowed"):
        validate_url("file:///etc/passwd")

    with pytest.raises(ValueError, match="URL scheme 'ftp' is not allowed"):
        validate_url("ftp://example.com/file.txt")

    with pytest.raises(ValueError, match="URL scheme '' is not allowed"):
        validate_url("/local/path/to/file")


def test_validate_url_custom_schemes() -> None:
    """Test that allowed schemes can be customized."""
    # Should pass
    validate_url("ftp://example.com", allowed_schemes=("ftp",))

    # Should fail with default schemes
    with pytest.raises(ValueError, match="URL scheme 'ftp' is not allowed"):
        validate_url("ftp://example.com")

    # Should fail when restricting allowed schemes
    with pytest.raises(ValueError, match="URL scheme 'http' is not allowed"):
        validate_url("http://example.com", allowed_schemes=("https",))
