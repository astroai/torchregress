## 2025-03-03 - [Fix SSRF risk by removing env var security bypass]
**Vulnerability:** A vulnerability was found in `torchregress/utils/security.py` where an environment variable (`TORCHREGRESS_SECURITY_ALLOW_FILE_URL`) could be used to bypass URL scheme checks. This could allow an attacker to bypass the checks and potentially access local files (SSRF/LFI).
**Learning:** Security validations should not rely on environment variables to bypass checks. Internal tools or tests that require relaxed constraints must explicitly pass the allowed overrides via arguments (e.g., `allowed_schemes=("http", "https", "file")`) to prevent SSRF or Local File Inclusion vulnerabilities.
**Prevention:** To prevent this, environment variables shouldn't control security boundaries in library code. Instead, use explicit arguments for overriding behavior, primarily for tools or tests.
## 2026-03-05 - [Fix Zip Slip vulnerability in examples]\n**Vulnerability:** Found a Zip Slip vulnerability in `examples/imdb_wiki_age_regression.py` where `zip_ref.extractall(DATA_DIR)` was used without validating that the files being extracted resolve to a path within the target directory. This could lead to a path traversal attack if a malicious zip file were used.\n**Learning:** When using `zipfile.ZipFile.extractall`, explicitly validate that all extracted members resolve to paths within the intended target directory to prevent path traversal (Zip Slip) vulnerabilities.\n**Prevention:** Use `os.path.commonpath([target_dir, member_path]) == target_dir` to enforce this boundary securely.
<<<<<<< HEAD
## 2026-03-05 - [Fix insecure deserialization via pandas.read_pickle]
**Vulnerability:** Found insecure deserialization vulnerabilities in `examples/photoz_benchmark_comparison.py` and `tools/photoz_nnc_pipeline.py` where `pd.read_pickle` was used to load datasets from paths that could potentially be user-provided. This could allow an attacker to achieve Arbitrary Code Execution (ACE/RCE) via a maliciously crafted pickle file.
**Learning:** Python's pickle module and functions that wrap it (like `pandas.read_pickle`) are inherently unsafe and should never be used to load untrusted data.
**Prevention:** To prevent this, do not support `.pkl` or `.pickle` formats for datasets or other external inputs. Enforce the use of safe, text-based data formats such as CSV, JSON, or JSONL instead.
=======

## 2026-03-08 - [Fix Insecure Deserialization via pd.read_pickle]
**Vulnerability:** Found an insecure deserialization vulnerability in `examples/photoz_benchmark_comparison.py` where `pd.read_pickle(dataset_path)` was used to load a user-supplied dataset path. An attacker could provide a malicious `.pkl` file, resulting in arbitrary code execution.
**Learning:** `pandas.read_pickle` relies on Python's `pickle` module, which is unsafe for untrusted data. Allowing users to specify pickle files as input paths in scripts opens the door to arbitrary code execution (ACE/RCE).
**Prevention:** Remove support for loading untrusted `.pkl` or `.pickle` files via user input. Use safer data formats like CSV, JSON, or Parquet for dataset loading.
>>>>>>> 9f35520 (🛡️ Sentinel: [CRITICAL] Fix Insecure Deserialization via pd.read_pickle)
