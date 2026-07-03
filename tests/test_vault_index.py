
import pytest

from synapse_mcp.wiki.vault_index import VaultIndex


@pytest.fixture
def temp_vault(tmp_path):
    """Create a temporary vault structure."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "wiki").mkdir()
    (vault / "raw").mkdir()
    return vault


@pytest.mark.asyncio
async def test_vault_index_initialization(temp_vault):
    """Test that VaultIndex tables and gitignore are created correctly."""
    index = VaultIndex(temp_vault)
    await index.initialize()

    # Check that .synapse directory was created
    assert (temp_vault / ".synapse").exists()

    # Check that .gitignore was updated
    gitignore = temp_vault / ".gitignore"
    assert gitignore.exists()
    content = gitignore.read_text()
    assert ".synapse/" in content

    # Check that connection can be opened and executes tables check
    conn = index._get_connection()
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [r[0] for r in res]
    assert "pages" in tables
    assert "link_graph" in tables
    index.close()


@pytest.mark.asyncio
async def test_vault_index_sync_and_query(temp_vault):
    """Test that syncing indexed pages works properly."""
    index = VaultIndex(temp_vault)
    await index.initialize()

    # Create dummy files
    page1_path = temp_vault / "wiki" / "page1.md"
    page1_path.write_text(
        "---\n"
        "summary: First page\n"
        "tags: [concept, test]\n"
        "created: 2026-06-04T12:00:00Z\n"
        "---\n"
        "This is page 1 body. It links to [[page2]] and [[page3]]."
    )

    page2_path = temp_vault / "wiki" / "page2.md"
    page2_path.write_text(
        "---\n"
        "summary: Second page\n"
        "tags: [entity]\n"
        "---\n"
        "This is page 2 body. It has a ## Connections section:\n"
        "- [[page1]]"
    )

    # Run sync
    res = await index.sync()
    assert res.added == 2
    assert res.updated == 0
    assert res.deleted == 0
    assert res.total == 2

    # Query page meta
    meta1 = await index.get_page_meta("wiki/page1.md")
    assert meta1 is not None
    assert meta1["name"] == "page1"
    assert meta1["summary"] == "First page"
    assert meta1["tags"] == ["concept", "test"]
    assert meta1["created"] == "2026-06-04T12:00:00Z"
    assert meta1["word_count"] == 11

    # Query list_pages
    list_res = await index.list_pages(subdir="wiki")
    assert list_res["total"] == 2
    assert len(list_res["pages"]) == 2

    # Query with tag filter
    list_tag = await index.list_pages(subdir="wiki", tag="concept")
    assert list_tag["total"] == 1
    assert list_tag["pages"][0]["name"] == "page1"

    # Query search
    search_res = await index.search(query="body page 1")
    assert len(search_res) == 2
    assert search_res[0]["name"] == "page1"
    assert "body" in search_res[0]["excerpt"]

    # Test incremental sync: modify page 1
    page1_path.write_text(
        "---\n"
        "summary: First page modified\n"
        "tags: [concept, new-tag]\n"
        "---\n"
        "New content."
    )

    res2 = await index.sync()
    assert res2.added == 0
    assert res2.updated == 1
    assert res2.deleted == 0
    assert res2.total == 2

    meta1_mod = await index.get_page_meta("wiki/page1.md")
    assert meta1_mod["summary"] == "First page modified"
    assert meta1_mod["tags"] == ["concept", "new-tag"]

    # Test deletion sync
    page2_path.unlink()
    res3 = await index.sync()
    assert res3.added == 0
    assert res3.updated == 0
    assert res3.deleted == 1
    assert res3.total == 1

    meta2 = await index.get_page_meta("wiki/page2.md")
    assert meta2 is None

    index.close()


@pytest.mark.asyncio
async def test_vault_index_lint(temp_vault):
    """Test SQL-based linting functionality."""
    index = VaultIndex(temp_vault)
    await index.initialize()

    # Create dummy files
    # page1 has valid links to page2, but page2 doesn't exist
    page1_path = temp_vault / "wiki" / "page1.md"
    page1_path.write_text(
        "---\n"
        "summary: Orphan page 1\n"
        "---\n"
        "This links to [[page2]] which is broken."
    )

    # page3 has malformed yaml
    page3_path = temp_vault / "wiki" / "page3.md"
    page3_path.write_text(
        "---\n" "title: Page 3\n" "malformed: : nested\n" "---\n" "Some body."
    )

    await index.sync()

    report = await index.lint()
    assert report["total_pages"] == 2
    assert report["knowledge_pages"] == 2

    # page1 and page3 are both orphans because they have no inbound links
    assert "page1" in report["orphan_pages"]
    assert "page3" in report["orphan_pages"]

    # Broken links
    assert len(report["broken_links"]) == 1
    assert report["broken_links"][0]["source"] == "wiki/page1.md"
    assert report["broken_links"][0]["target"] == "page2"

    # Invalid frontmatter
    assert len(report["invalid_frontmatter"]) == 1
    assert report["invalid_frontmatter"][0]["page"] == "wiki/page3.md"
    assert "mapping values are not allowed" in report["invalid_frontmatter"][0]["error"]

    index.close()


@pytest.mark.asyncio
async def test_vault_index_write_through(temp_vault):
    """Test write-through (upsert_page, remove_page) operations."""
    index = VaultIndex(temp_vault)
    await index.initialize()

    page1_path = temp_vault / "wiki" / "page1.md"
    page1_path.write_text("---\n" "summary: Hello\n" "---\n" "Body prose.")

    # Direct upsert (without full sync)
    await index.upsert_page("wiki/page1.md")

    meta = await index.get_page_meta("wiki/page1.md")
    assert meta is not None
    assert meta["summary"] == "Hello"

    # Update and upsert again
    page1_path.write_text("---\n" "summary: Hello World\n" "---\n" "Body prose.")
    await index.upsert_page("wiki/page1.md")
    meta = await index.get_page_meta("wiki/page1.md")
    assert meta["summary"] == "Hello World"

    # Direct remove
    await index.remove_page("wiki/page1.md")
    meta = await index.get_page_meta("wiki/page1.md")
    assert meta is None

    index.close()


@pytest.mark.asyncio
async def test_vault_index_multiprocess_concurrency(temp_vault):
    """Two processes can use the SQLite/WAL index simultaneously.

    The old version of this test asserted an in-memory *fallback* kicked
    in when another process held a DuckDB write lock. The index migrated
    to SQLite in WAL mode, where cross-process concurrency is supported
    directly — so the correct assertion now is that a second process
    holding an open connection does NOT degrade this process's index.
    """
    import subprocess
    import sys

    db_path = temp_vault / ".synapse" / "vault_index.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Pre-initialize the database so it exists
    index_writer = VaultIndex(temp_vault)
    await index_writer.initialize()
    index_writer.close()

    # 2. Spawn a separate process holding an open connection that has
    #    recently written (WAL allows concurrent readers/writers).
    lock_code = (
        "import sqlite3, time, sys\n"
        "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
        "conn.execute('PRAGMA journal_mode=WAL')\n"
        "conn.execute('CREATE TABLE IF NOT EXISTS hold_table (val INTEGER)')\n"
        "conn.execute('INSERT INTO hold_table VALUES (1)')\n"
        "print('READY', flush=True)\n"
        "time.sleep(10)\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", lock_code, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        ready_line = proc.stdout.readline().strip()
        assert ready_line == "READY", (
            f"Subprocess failed to start: {proc.stderr.read()}"
        )

        # 3. Our index must initialize and stay fully functional — no
        #    fallback, real reads AND writes against the shared DB.
        index = VaultIndex(temp_vault)
        await index.initialize()
        assert index._is_fallback is False

        res = await index.list_pages(subdir="wiki")
        assert isinstance(res, dict)
        assert "pages" in res

        page_path = temp_vault / "wiki" / "some_page.md"
        page_path.write_text("---\nsummary: Concurrent write\n---\nHello WAL")

        sync_res = await index.sync()
        assert sync_res.added == 1
        assert sync_res.total == 1

        meta = await index.get_page_meta("wiki/some_page.md")
        assert meta is not None
        assert meta["summary"] == "Concurrent write"

        await index.remove_page("wiki/some_page.md")
        assert await index.get_page_meta("wiki/some_page.md") is None

        index.close()
    finally:
        proc.terminate()
        proc.wait()
