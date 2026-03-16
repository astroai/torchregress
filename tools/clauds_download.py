"""Download CLAUDS HSCpipe and optionally SExtractor FITS from CADC."""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from torchregress.utils.security import validate_url

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "clauds_specz" / "raw"
BASE_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/clauds/desprez/PublicRelease"
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB stream
MIN_FREE_GB = 15  # warn if less than this (multi-GB FITS)

HSCPIPE_FILES = [
    "COSMOS-HSCpipe-Phosphoros.fits",
    "DEEP23-HSCpipe-Phosphoros.fits",
    "XMMLSS-HSCpipe-Phosphoros.fits",
    "ELAIS-N1-HSCpipe-Phosphoros.fits",
]

SEXTRACTOR_FILES = [
    "COSMOS_11bands-SExtractor-Lephare.fits",
    "COSMOS_6bands-SExtractor-Lephare.fits",
    "XMMLSS_11bands-SExtractor-Lephare.fits",
    "XMMLSS_6bands-SExtractor-Lephare.fits",
    "ELAIS-N1_6bands-SExtractor-Lephare.fits",
    "DEEP2-3_6bands-SExtractor-Lephare.fits",
]


def _file_md5(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute MD5 of file as 32-char hex (chunked for large files)."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _expected_md5_from_headers(url: str) -> str | None:
    """
    Perform HEAD request and return expected MD5 as 32-char hex if present in response headers.
    Checks Content-MD5 (Base64), ETag (if 32 hex chars), X-Checksum-Md5, X-Content-Md5.
    """
    url = validate_url(url, allowed_schemes=("http", "https"))
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=30) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
    except Exception:
        return None

    # Content-MD5: Base64-encoded 128-bit digest (RFC 1864)
    raw = headers.get("content-md5")
    if raw:
        try:
            digest_bytes = base64.b64decode(raw.strip())
            if len(digest_bytes) == 16:
                return digest_bytes.hex()
        except Exception:
            pass

    # ETag: sometimes quoted MD5 in hex, e.g. "abc123..." or W/"abc123..."
    raw = headers.get("etag")
    if raw:
        s = raw.strip().strip('"').lower()
        if s.startswith("w/"):
            s = s[2:].strip('"')
        if len(s) == 32 and all(c in "0123456789abcdef" for c in s):
            return s

    # Custom headers some servers use
    for key in ("x-checksum-md5", "x-content-md5", "x-md5-checksum"):
        raw = headers.get(key)
        if raw:
            s = raw.strip().lower()
            if len(s) == 32 and all(c in "0123456789abcdef" for c in s):
                return s
            # might be Base64
            try:
                digest_bytes = base64.b64decode(s)
                if len(digest_bytes) == 16:
                    return digest_bytes.hex()
            except Exception:
                pass

    # Content-Digest (draft) or Digest (CADC): e.g. md5=base64 or md5=:base64:
    for key in ("content-digest", "digest"):
        raw = headers.get(key)
        if raw and "md5=" in raw.lower():
            try:
                # "md5=wH4tbcwtYFA6ojSPryPc7A==" or "md5=:base64:"
                parts = raw.split("=", 1)[1].strip().strip(":").strip()
                digest_bytes = base64.b64decode(parts)
                if len(digest_bytes) == 16:
                    return digest_bytes.hex()
            except Exception:
                pass

    return None


def _expected_fits_size(path: Path) -> int | None:
    """Return expected FITS table size in bytes from header (data_offset + NAXIS1*NAXIS2), or None if unreadable."""
    try:
        from astropy.io import fits

        with fits.open(path) as hdul:
            if len(hdul) < 2:
                return None
            hdu = hdul[1]
            naxis1 = int(hdu.header.get("NAXIS1", 0))
            naxis2 = int(hdu.header.get("NAXIS2", 0))
            start = getattr(hdu, "_data_offset", 0)
            return start + naxis1 * naxis2
    except Exception:
        return None


def _free_gb(path: Path) -> float:
    """Approximate free space in GB for the filesystem containing path."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024**3)
    except OSError:
        return float("inf")


def _wget_available() -> bool:
    try:
        subprocess.run(
            ["wget", "--version"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _download_via_wget(url: str, target: Path) -> None:
    """Download with wget (resume, progress, robust for multi-GB files)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    url = validate_url(url, allowed_schemes=("http", "https"))
    try:
        subprocess.run(
            [
                "wget",
                "-c",  # resume partial
                "--show-progress",
                "-O",
                str(target),
                url,
            ],
            check=True,
            timeout=3600,
        )
        if target.exists() and target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            raise OSError("wget produced 0-byte file")
    except subprocess.CalledProcessError as e:
        # Only remove partial/0-byte file so we don't delete existing data on transient errors
        if target.exists() and target.stat().st_size == 0:
            target.unlink(missing_ok=True)
        raise OSError(f"wget failed: {e}") from e


def _download_to_path(url: str, target: Path) -> None:
    """Stream download to target; remove partial/0-byte file on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    url = validate_url(url, allowed_schemes=("http", "https"))
    try:
        with urlopen(url, timeout=60) as response:
            code = getattr(response, "status", None) or getattr(response, "getcode", lambda: None)()
            if code is not None and code != 200:
                raise OSError(f"HTTP {code}")
            with target.open("wb") as out:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
    except Exception:
        if target.exists() and target.stat().st_size >= 0:
            target.unlink(missing_ok=True)
        raise


def download_clauds(
    *,
    output_dir: Path = DEFAULT_RAW_DIR,
    hscpipe: bool = True,
    sextractor: bool = False,
    overwrite: bool = False,
    use_wget: bool | None = None,
) -> dict[str, str]:
    """Download CLAUDS FITS to output_dir. Returns paths and status per file."""
    output_dir = Path(output_dir)
    results: dict[str, str] = {}
    files = []
    if hscpipe:
        files.extend(HSCPIPE_FILES)
    if sextractor:
        files.extend(SEXTRACTOR_FILES)

    use_wget = use_wget if use_wget is not None else _wget_available()
    if use_wget and not _wget_available():
        use_wget = False
        print("  Warning: wget not found; using Python urllib.")

    free_gb = _free_gb(output_dir)
    if free_gb < MIN_FREE_GB:
        print(
            f"  Warning: low disk space ({free_gb:.1f} GB free). Downloads may fail or run out of space."
        )

    download_fn = _download_via_wget if use_wget else _download_to_path
    for filename in files:
        target = output_dir / filename
        url = f"{BASE_URL.rstrip('/')}/{filename}"

        # Prefer MD5 from response headers for integrity; fall back to FITS size when no header
        expected_md5 = _expected_md5_from_headers(url)

        if target.exists() and not overwrite:
            size = target.stat().st_size
            if size == 0:
                pass  # retry
            elif expected_md5:
                if _file_md5(target) == expected_md5:
                    results[filename] = "existing"
                    continue
                results[filename] = "checksum mismatch (re-downloading)"
            else:
                expected_size = _expected_fits_size(target) if filename in HSCPIPE_FILES else size
                if expected_size is not None and size >= expected_size:
                    results[filename] = "existing"
                    continue
                if size > 0 and expected_size is not None and size < expected_size:
                    results[filename] = "incomplete (re-downloading)"

        try:
            download_fn(url, target)
            if expected_md5:
                actual = _file_md5(target)
                if actual != expected_md5:
                    if target.stat().st_size == 0:
                        target.unlink(missing_ok=True)
                    raise OSError(
                        f"MD5 mismatch for {filename}: got {actual}, expected {expected_md5}"
                    )
            results[filename] = "downloaded"
        except Exception as e:
            results[filename] = f"error: {e!s}"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CLAUDS photometry FITS from CADC (HSCpipe and/or SExtractor)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to write FITS files.",
    )
    parser.add_argument(
        "--no-hscpipe",
        action="store_true",
        help="Skip HSCpipe files (default: download HSCpipe).",
    )
    parser.add_argument(
        "--sextractor",
        action="store_true",
        help="Also download SExtractor/Le Phare FITS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--no-wget",
        action="store_true",
        help="Use Python urllib instead of wget (wget is default when available).",
    )
    args = parser.parse_args()
    results = download_clauds(
        output_dir=args.output_dir,
        hscpipe=not args.no_hscpipe,
        sextractor=args.sextractor,
        overwrite=args.overwrite,
        use_wget=not args.no_wget,
    )
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
