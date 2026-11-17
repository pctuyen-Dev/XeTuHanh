---
name: clean-code-handover
description: >-
  Refactor, clean up, and organize messy codebases by removing redundant/dead code,
  structuring file and module layouts, making functions easy to read and locate,
  and generating a formal Handover Report (Biên bản bàn giao). Use when the user
  asks to clean up code, remove unused/redundant code, reorganize project/file
  structure, improve readability, or produce a handover document.
---

# Clean Code & Handover Workflow

Use this skill to transform messy, disorganized code into a clean, maintainable codebase with a clear layout, easy-to-find functions, zero dead code, and a complete **Biên bản bàn giao** (Handover Report).

## Core Principles

1. **Behavior First**: Never change functional behavior silently. Capture baseline checks/tests before refactoring and re-verify after edits.
2. **Zero Redundancy**: Remove unused imports, dead variables, commented-out code blocks, duplicate logic, and leftover debug/test snippets.
3. **Predictable Layout**: Every folder and file must follow a consistent, top-down structure so any developer can locate code in seconds.
4. **Readable, Findable Functions**: Keep functions short, single-purpose, shallowly nested, clearly named (`verb_noun`), and logically ordered.
5. **Mandatory Handover**: Always conclude with a **Biên bản bàn giao** using [handover_template.md](./resources/handover_template.md).

---

## Step-by-Step Procedure

### Step 1: Audit & Baseline
1. Inspect the target files/folders and map out:
   - Entry points, core modules, hardware/I/O drivers, utilities, configs, and assets.
   - Dead/redundant code (unused imports/functions/variables, duplicate blocks, commented-out old code, magic numbers).
   - Structural issues (oversized files, mixed responsibilities, hard-to-find functions).
2. Run available syntax checks, linters, or tests to establish a baseline before editing.

### Step 2: Remove Redundant & Dead Code
- Delete unused imports, variables, parameters, and unreachable branches.
- Remove commented-out code blocks and temporary debug prints (keep intentional logging).
- Deduplicate repeated logic into shared helper functions.
- Extract scattered magic numbers/strings into named constants or a config section.

### Step 3: Organize File & Module Layout
Apply a consistent top-to-bottom layout inside each source file:
1. **Header / Docstring**: Brief description of the file's responsibility.
2. **Imports**: Grouped cleanly (standard library -> third-party -> local modules).
3. **Constants & Configuration**: Named constants, pins, thresholds, paths.
4. **Data Structures / Classes**: Models, state containers, hardware wrappers.
5. **Public / High-Level Functions**: Main workflow and exported functions (ordered top-down: caller before callee).
6. **Private / Helper Functions**: Internal helpers prefixed or grouped clearly below public functions.
7. **Entry Point**: `if __name__ == "__main__":` (Python) or `setup()` / `loop()` (Arduino/C++) at the bottom (or standard lifecycle order).

For multi-file projects, group files logically by responsibility (e.g., `config/`, `drivers/`, `vision/`, `control/`, `utils/`) when refactoring project structure, and avoid breaking existing imports/paths without updating all references.

### Step 4: Make Functions Easy to Read & Find
- **Naming**: Use clear `verb_noun` names that describe *what* the function does (e.g., `read_ultrasonic_distance`, `detect_traffic_sign`).
- **Size & Nesting**: Keep functions focused on one task; use early returns to avoid deep `if/else` nesting.
- **Documentation**: Add concise docstrings/comments stating purpose, parameters, return values, and units/constraints where non-obvious.
- **Findability**: Group related functions under clear section headers (e.g., `# --- Motor Control ---`) and list key functions in the Handover Report's function directory.

### Step 5: Verify Integrity
1. Re-run syntax checks, static analysis, and tests after the final edit:
   - Python: `python -m py_compile <files>` and any `pytest` / linter available.
   - C/C++ / Arduino: compile/verify check if toolchain is available.
2. Confirm no broken imports, missing references, or accidental behavior changes.
3. Explicitly note any parts that could not be executed (e.g., physical hardware/ESP32/camera not connected).

### Step 6: Generate Biên Bản Bàn Giao (Handover Report)
Create a handover document (e.g., `BIEN_BAN_BAN_GIAO.md` in the project root or target folder, unless the user specifies another path) following [handover_template.md](./resources/handover_template.md).

Ensure the report accurately records:
- Only observed files, symbol names, removed items, and actual command/test results.
- A clear **Function Directory (Danh mục hàm chính)** mapping `File -> Function -> Purpose` so the recipient can find any function immediately.
- Remaining risks, hardware dependencies, and recommended next steps.
