# Plan: Phase 2 — Wiki Bridge Resilience

**Goal:** Harden the `WikiAdapter` against manual user file edits in the Obsidian vault and ensure the index remains consistent.

## 02.1: Robust File Handling & Error Mapping
- [ ] **Task:** Refactor `WikiAdapter.read_page` to handle race conditions and return structured errors.
- [ ] **Task:** Audit `WikiAdapter.write_page` for safety (ensure parent directories exist, etc.).
- [ ] **Task:** Implement specific exception types for Wiki operations (e.g., `WikiPageNotFoundError`).

## 02.2: Resilient Index Management
- [ ] **Task:** Update `WikiAdapter.update_index` to detect and remove orphaned links.
- [ ] **Task:** Add a "deep refresh" mode to `wiki_update_index` that verifies file existence on disk.
- [ ] **Task:** Ensure `wiki_list_pages` handles missing files during iteration (in case of concurrent deletion).

## 02.3: Verification & TDD
- [ ] **Task:** Create `tests/test_wiki_resilience.py`.
- [ ] **Task:** Implement test cases for:
    - Renaming a page externally and then trying to read it via the old path.
    - Deleting a page externally and running `wiki_update_index`.
    - Concurrent read/delete operations.
- [ ] **Task:** Verify that `server.py` tool handlers propagate these new errors as user-friendly messages.

## 02.4: Roadmap Update
- [ ] **Task:** Update `ROADMAP.md` and `STATE.md` with Phase 2 progress.
