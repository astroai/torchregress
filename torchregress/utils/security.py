"""
Security utilities.
"""

import os
from urllib.parse import urlparse


def validate_url(url: str, allowed_schemes: tuple = ("http", "https")) -> str:
    """
    Validate that a URL uses an allowed scheme.

    Args:
        url: The URL string to validate.
        allowed_schemes: A tuple of allowed URL schemes (default: http, https).

    Returns:
        The valid URL string.

    Raises:
        ValueError: If the URL scheme is not allowed.
    """
    # For testing and local dev with trusted manifests
    if os.environ.get("TORCHREGRESS_SECURITY_ALLOW_FILE_URL") == "1":
        allowed_schemes = allowed_schemes + ("file",)

    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. " f"Allowed schemes: {allowed_schemes}"
        )
    return url
