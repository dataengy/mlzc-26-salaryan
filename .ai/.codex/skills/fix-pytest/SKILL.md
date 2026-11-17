---
name: fix-pytest
description: Diagnose and fix pytest discovery, import, invocation, and environment errors in Python projects.
---

# Fix Pytest

Use this skill when pytest cannot be found, collects no tests, cannot import a test module, or reports confusing collection/configuration errors.

- Reproduce the exact command and capture the interpreter, working directory, pytest rootdir, config file, and collection traceback.
- Distinguish shell PATH problems from Python import problems. Check the active shell and interpreter (`command -v pytest`, `python -m pytest --version`, `sys.executable`, and `sys.path`). Prefer the project environment's `python -m pytest` when the `pytest` executable is missing.
- For imports, inspect package layout and pytest's import mode. Avoid relying on ambiguous top-level names such as `tests` unless the repository package and root path are explicit. Prefer a minimal project config fix or unambiguous relative/local imports.
- Verify with `pytest --collect-only` on the affected test before running expensive tests. Run the requested test only when useful and authorized.
- Keep changes focused; don't change global shell configuration to fix a project-local pytest import error.
