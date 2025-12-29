## Installation (with [uv](https://github.com/astral-sh/uv))

`uv` is a fast Python package/environment manager developed by Astral.

### 1. Install `uv`
If you don’t already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or run `source ~/.bashrc` / `source ~/.zshrc`).

Check installation:
```bash
uv --version
```

---

### 2. Clone the repository

```bash
git clone https://github.com/your-username/nufft_biot.git
cd nufft_biot
```

---

### 3. Create and sync environment

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml`.

To activate the environment:
```bash
uv run python
```
or for any command:
```bash
uv run <command>
```

Example:
```bash
uv run pytest -v
```

## Run tests

```bash
uv run pytest -v
```
