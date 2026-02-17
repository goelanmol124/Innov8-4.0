"""
AST-based security scanner for participant submissions.

Scans every .py file in a submission for banned imports and dangerous calls
before any code is executed. If any violation is found the team is
disqualified without running their code.

Allowed libraries are whitelisted explicitly; everything else is blocked.
"""

import ast
from pathlib import Path

# ─── Allowlist ────────────────────────────────────────────────────────────────
# Top-level module names participants are permitted to import.
ALLOWED_TOP_LEVEL = {
    # numerics / ML
    "numpy", "pandas", "sklearn", "scipy",
    # stdlib — safe utilities
    "math", "random", "statistics", "numbers",
    "collections", "itertools", "functools", "operator",
    "heapq", "queue", "bisect",
    "typing", "types", "abc", "dataclasses", "enum",
    "copy", "warnings", "pprint",
    "time", "datetime",
    "json", "re", "string",
    "pathlib",         # read-only use is fine; open() is blocked separately
}

# ─── Banlist (explicit) ───────────────────────────────────────────────────────
# Any import whose top-level module matches is an instant disqualification.
BANNED_TOP_LEVEL = {
    # OS / shell access
    "os", "sys", "subprocess", "shutil", "tempfile",
    "pty", "tty", "nis", "syslog", "pwd", "grp",
    "resource", "termios", "signal", "fcntl",
    # Networking
    "socket", "socketserver", "ssl",
    "requests", "httpx", "aiohttp", "urllib", "urllib3",
    "http", "ftplib", "smtplib", "poplib", "imaplib",
    "xmlrpc", "nntplib", "telnetlib",
    # Async / multiprocessing (could spawn processes)
    "asyncio", "concurrent", "multiprocessing", "threading",
    "_thread", "greenlet", "gevent",
    # Introspection / code execution
    "ast", "dis", "py_compile", "compileall",
    "gc", "ctypes", "cffi",
    "inspect", "importlib", "pkgutil", "imp",
    "builtins", "linecache", "tokenize",
    # Serialisation (can exec arbitrary code via pickle)
    "pickle", "pickletools", "shelve", "marshal",
    # Misc dangerous
    "code", "codeop", "readline", "rlcompleter",
    "platform", "struct",
}

# ─── Banned bare function calls ───────────────────────────────────────────────
BANNED_CALLS = {
    "eval", "exec", "compile",
    "__import__",
    "open",          # file I/O
    "breakpoint",
    "input",
    "vars", "dir",   # introspection helpers
    "getattr",       # can be used to bypass attribute bans
    "setattr", "delattr",
    "globals", "locals",
}

# ─── Banned attribute accesses (dunder gymnastics) ───────────────────────────
BANNED_ATTRS = {
    "__closure__", "__code__", "__globals__", "__builtins__",
    "__subclasses__", "__mro__", "__dict__", "__class__",
    "__reduce__", "__reduce_ex__",
    "mro",
    # subprocess-related attributes
    "system", "popen", "Popen", "run", "call",
    "check_output", "check_call", "getoutput", "getstatusoutput",
}


# ─── Visitor ─────────────────────────────────────────────────────────────────

class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"{self.filename}:{node.lineno}: {msg}")

    # ── Import checks ────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BANNED_TOP_LEVEL:
                self._flag(node, f"banned import: '{alias.name}'")
            elif top not in ALLOWED_TOP_LEVEL:
                self._flag(node, f"disallowed import: '{alias.name}' (not in allowlist)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        top = node.module.split(".")[0]
        if top in BANNED_TOP_LEVEL:
            self._flag(node, f"banned import: 'from {node.module} import ...'")
        elif top not in ALLOWED_TOP_LEVEL:
            self._flag(node, f"disallowed import: 'from {node.module}' (not in allowlist)")
        self.generic_visit(node)

    # ── Call checks ──────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        # eval(), exec(), open(), __import__(), …
        if isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                self._flag(node, f"banned call: '{node.func.id}()'")

        # obj.system(), subprocess.Popen(), etc.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in BANNED_ATTRS:
                self._flag(node, f"banned method call: '.{node.func.attr}()'")

        self.generic_visit(node)

    # ── Attribute access checks ──────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_ATTRS:
            self._flag(node, f"banned attribute access: '.{node.attr}'")
        self.generic_visit(node)


# ─── Public API ───────────────────────────────────────────────────────────────

def check_file(filepath: Path) -> list[str]:
    """
    AST-scan a single .py file.

    Returns a list of violation strings (empty = clean).
    """
    try:
        src = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [f"{filepath}: cannot read file: {e}"]

    try:
        tree = ast.parse(src, filename=str(filepath))
    except SyntaxError as e:
        return [f"{filepath}:{e.lineno}: SyntaxError: {e.msg}"]

    visitor = _SecurityVisitor(str(filepath.name))
    visitor.visit(tree)
    return visitor.violations


def check_submission(team_dir: Path) -> list[str]:
    """
    AST-scan every .py file found recursively under team_dir.

    Returns all violations across all files (empty list = submission is clean).
    """
    all_violations: list[str] = []
    for py_file in sorted(team_dir.rglob("*.py")):
        all_violations.extend(check_file(py_file))
    return all_violations


# ─── CLI helper ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = list(target.rglob("*.py")) if target.is_dir() else [target]

    found = False
    for f in files:
        violations = check_file(f)
        if violations:
            found = True
            for v in violations:
                print(f"VIOLATION  {v}")

    if not found:
        print("All clear — no violations found.")
    else:
        sys.exit(1)
