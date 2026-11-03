# setup the project with pytorch gpu
_Exported on 11/03/2026 at 12:47:02 GMT-6 from Claude via WayLog_


**User**

setup the project with pytorch gpu

---


**Claude**

Your RTX 5090's driver supports CUDA 13.3, and torch 2.15.0 ships cu132 and cu134 wheels. cu134 needs a newer driver, so I'll use cu132. I'll pin it for Windows and Linux; macOS keeps using PyPI.

---


**Claude**

CUDA is working. Running the tests and the lint check next.

---


**Claude**

Training turns on `torch.use_deterministic_algorithms(True)`. On CUDA, that setting usually makes cuBLAS calls fail unless `CUBLAS_WORKSPACE_CONFIG` is set, so I'll check whether it happens here:

---


**Claude**

The forward and backward pass both work on the GPU in deterministic mode. Updating the README's setup note next:

---


**Claude**

PyTorch now runs on your RTX 5090. `uv sync` installs `torch 2.15.0+cu132`, which reports CUDA available and has kernels for the 5090's architecture (`sm_120`). A matrix multiply and a model forward/backward pass both ran on the GPU in the training script's deterministic mode. All 37 tests pass and `ruff check` is clean.

**What I changed:**
- **[pyproject.toml](pyproject.toml):** added the PyTorch CUDA 13.2 wheel index and pointed torch at it for Windows and Linux. macOS still gets torch from PyPI. I chose CUDA 13.2 because your driver supports up to 13.3, so the 13.4 wheels wouldn't run on it.
- **[uv.lock](uv.lock):** regenerated with `uv lock`.
- **[README.md](README.md):** replaced the old CUDA note with how the GPU setup works and how to check it.

Training still defaults to CPU, so pass `--device cuda`:
```bash
uv run python -m asd_contextpose.train --device cuda ...
```

Your GPU was almost full when I checked: about 31.9 of 32.6 GB in use at 89% load, with another `python.exe` (PID 30836) holding it. If training hits out-of-memory errors, that process is the likely cause.

I haven't committed anything.

---
