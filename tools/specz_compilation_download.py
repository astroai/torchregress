"""Download or locate COSMOS Spec-z Compilation FITS (cosmosastro/speczcompilation)."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

from torchregress.utils.security import validate_url

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "clauds_specz" / "specz_compilation"
DEFAULT_SPECZ_UNIQUE_FILENAME = "specz_compilation_COSMOS_DR1.1_unique.fits"

# GitHub repo cosmosastro/speczcompilation: raw gives LFS pointer; media serves actual file.
DEFAULT_SPECZ_URL = (
    "https://media.githubusercontent.com/media/cosmosastro/speczcompilation/main/"
    "specz_compilation/specz_compilation_COSMOS_DR1.1_unique.fits"
)
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


def _download_to_path(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = validate_url(url, allowed_schemes=("http", "https"))
    with urlopen(url, timeout=60) as response, target.open("wb") as out:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)


def ensure_specz_compilation(
    *,
    output_dir: Path = DEFAULT_RAW_DIR,
    url: str | None = None,
    local_path: Path | None = None,
    filename: str = DEFAULT_SPECZ_UNIQUE_FILENAME,
    overwrite: bool = False,
) -> Path | None:
    """
    Ensure the spec-z compilation FITS is available under output_dir.

    - If local_path is set and exists, copy to output_dir/filename (or return local_path).
    - If url is set, download to output_dir/filename.
    - If output_dir/filename already exists and not overwrite, return it.
    Otherwise return None (user must provide URL or local path).
    """
    output_dir = Path(output_dir)
    target = output_dir / filename

    if target.exists() and not overwrite:
        return target

    if local_path is not None:
        src = Path(local_path)
        if src.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(src, target)
            return target
        if src.is_dir() and (src / filename).exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(src / filename, target)
            return target
        return None

    if url:
        _download_to_path(url, target)
        return target

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or register COSMOS Spec-z Compilation FITS (DR1.1)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for the unique FITS file.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to download specz_compilation_COSMOS_DR1.1_unique.fits (e.g. Zenodo).",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=None,
        help="Local path to the FITS file or directory containing it.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_SPECZ_UNIQUE_FILENAME,
        help="FITS filename (default: specz_compilation_COSMOS_DR1.1_unique.fits).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file.",
    )
    args = parser.parse_args()

    path = ensure_specz_compilation(
        output_dir=args.output_dir,
        url=args.url if args.url is not None else DEFAULT_SPECZ_URL,
        local_path=args.local_path,
        filename=args.filename,
        overwrite=args.overwrite,
    )
    if path is not None:
        print(f"Spec-z compilation FITS: {path}")
    else:
        print(
            "No FITS obtained. Provide --url or --local-path. "
            "Default is GitHub (cosmosastro/speczcompilation) LFS media URL."
        )


if __name__ == "__main__":
    main()
