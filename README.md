# PDR

## Environment Setup (uv)

Use [uv](https://github.com/astral-sh/uv) to manage the Python toolchain and virtual environment:

```bash
# install uv if it is not already available on your system
curl -Ls https://astral.sh/uv/install.sh | sh

# download the Python runtime you want to use (example: 3.12)
uv python install 3.12

# create the virtual environment and install dependencies declared in
# pyproject.toml / uv.lock
uv sync --python 3.12

# activate the environment (Linux/macOS)
source .venv/bin/activate
# or on Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

Populate `pyproject.toml` (or `uv.lock`) with the dependencies you need before running `uv sync`.
