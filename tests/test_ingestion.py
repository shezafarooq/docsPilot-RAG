import re
import unittest
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient

from app.chunking import chunk_document
from app.documents import extract_document
from app.vector_store import ensure_collection, sync_source


class WordTokenizer:
    """Small deterministic tokenizer so unit tests need no model download."""

    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    def num_special_tokens_to_add(self):
        return 2

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class TestDocuments(unittest.TestCase):
    def chunk_fixture(self, text):
        model = type("Model", (), {"tokenizer": WordTokenizer(), "max_seq_length": 256})()
        document = {"source_url": "https://example.com/docs", "title": "Example", "project": "docker",
                    "sections": [{"heading": "Explanation", "url": "https://example.com/docs#explanation", "text": text}]}
        return chunk_document(document, model)

    def test_definition_is_not_buried_in_an_unrelated_introductory_paragraph(self):
        introduction = " ".join(["This introduction describes a team installing several different tools on their development machines."] * 5)
        definition = "A container is an isolated process with all the files that it needs to run an application."
        chunks = self.chunk_fixture(introduction + "\n\n" + definition)
        self.assertTrue(any(chunk["text"] == definition for chunk in chunks))

    def test_code_keeps_its_blank_lines_and_preceding_explanation(self):
        code = "```yaml\nservices:\n  app:\n    volumes:\n      - data:/data\n\nvolumes:\n  data:\n```"
        chunks = self.chunk_fixture("Mount the named volume as follows.\n\n" + code)
        self.assertEqual(len(chunks), 1)
        self.assertIn(code, chunks[0]["text"])
        self.assertTrue(chunks[0]["text"].startswith("Mount the named volume"))

    def test_repeated_paragraphs_still_have_unique_point_ids(self):
        paragraph = "Containers provide isolated environments for applications and include the files needed to run those applications."
        chunks = self.chunk_fixture(paragraph + "\n\n" + paragraph)
        self.assertEqual(len(chunks), len({chunk["id"] for chunk in chunks}))

    def test_article_is_preferred_over_enclosing_main_with_sidebar(self):
        html = '''<main><div><ul><li>Get started</li><li>Guides</li></ul></div>
        <h1>Volumes</h1><article><p>Mount a named volume.</p></article></main>'''
        document = extract_document(html, "https://docs.docker.com/example/")
        self.assertEqual(document["title"], "Volumes")
        self.assertEqual(document["sections"][0]["text"], "Mount a named volume.")

    def test_article_excludes_navigation_and_preserves_code_and_source_anchor(self):
        html = '''<nav>Outside navigation</nav><article><h1>Compose</h1>
        <nav>Inside navigation</nav><h2 id="volumes">Volumes</h2>
        <p>Mount a directory.</p><pre>volumes:\n  - ./data:/app/data</pre></article>'''
        document = extract_document(html, "https://docs.docker.com/example/")
        section = document["sections"][0]
        self.assertEqual(section["url"], "https://docs.docker.com/example/#volumes")
        self.assertIn("volumes:\n  - ./data:/app/data", section["text"])
        self.assertNotIn("navigation", section["text"])

    def test_fragment_includes_child_sections_but_excludes_sibling_sections(self):
        html = '''<article><h1>Services</h1><h2 id="ports">Ports</h2><p>Exclude ports.</p>
        <h2 id="volumes">Volumes</h2><p>Include mounts.</p>
        <h3 id="syntax">Syntax</h3><p>Include syntax.</p>
        <h2 id="working-dir">Working directory</h2><p>Exclude directory.</p></article>'''
        document = extract_document(html, "https://docs.docker.com/example/#volumes")
        self.assertEqual([section["heading"] for section in document["sections"]], ["Volumes", "Syntax"])

    def test_missing_requested_section_fails_instead_of_indexing_wrong_content(self):
        with self.assertRaises(ValueError):
            extract_document("<article><h1>Other</h1><p>Unrelated.</p></article>", "https://docs.docker.com/example/#missing")

    def test_python_version_tabs_do_not_duplicate_examples(self):
        html = '''<article><h1>Body</h1><div class="tabbed-content">
        <div class="tabbed-block"><pre>keep_this()</pre></div>
        <div class="tabbed-block"><pre>old_variant()</pre></div></div></article>'''
        document = extract_document(html, "https://fastapi.tiangolo.com/tutorial/body/")
        self.assertIn("keep_this", document["sections"][0]["text"])
        self.assertNotIn("old_variant", document["sections"][0]["text"])

    def test_chunks_cover_document_fit_model_and_have_stable_ids(self):
        model = type("Model", (), {"tokenizer": WordTokenizer(), "max_seq_length": 64})()
        words = [f"word{number}" for number in range(180)]
        document = {"source_url": "https://example.com/docs", "title": "Example",
                    "project": "docker", "sections": [{"heading": "Section",
                    "url": "https://example.com/docs#section", "text": " ".join(words)}]}
        chunks = chunk_document(document, model, overlap=8)
        recovered = {word for chunk in chunks for word in chunk["text"].split()}
        self.assertEqual(recovered, set(words))
        self.assertTrue(all(chunk["token_count"] <= 64 for chunk in chunks))
        self.assertEqual([chunk["id"] for chunk in chunks], [chunk["id"] for chunk in chunk_document(document, model, overlap=8)])
        self.assertTrue(all(chunk["url"].endswith("#section") for chunk in chunks))


class TestIndexUpdates(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        ensure_collection(self.client, "docs", 3, "test-model")

    def tearDown(self):
        self.client.close()

    def chunk(self, source, text):
        return {"id": str(uuid5(NAMESPACE_URL, source + text)), "source_url": source,
                "url": source + "#section", "text": text}

    def test_reindex_is_idempotent_and_changed_source_keeps_other_documents(self):
        first = [self.chunk("https://example.com/a", "old")]
        second = [self.chunk("https://example.com/b", "other")]
        for _ in range(2):
            sync_source(self.client, "docs", first, [[1., 0., 0.]], "test-model")
        sync_source(self.client, "docs", second, [[0., 1., 0.]], "test-model")
        self.assertEqual(self.client.count("docs", exact=True).count, 2)
        replacement = [self.chunk("https://example.com/a", "new")]
        sync_source(self.client, "docs", replacement, [[0., 0., 1.]], "test-model")
        points, _ = self.client.scroll("docs")
        self.assertEqual({point.payload["text"] for point in points}, {"new", "other"})

    def test_different_model_or_vector_dimension_is_rejected(self):
        sync_source(self.client, "docs", [self.chunk("https://example.com/a", "text")], [[1., 0., 0.]], "test-model")
        with self.assertRaises(ValueError):
            ensure_collection(self.client, "docs", 4, "test-model")
        with self.assertRaises(ValueError):
            ensure_collection(self.client, "docs", 3, "different-model")

    def test_missing_embeddings_do_not_delete_existing_document(self):
        chunk = self.chunk("https://example.com/a", "text")
        sync_source(self.client, "docs", [chunk], [[1., 0., 0.]], "test-model")
        with self.assertRaises(ValueError):
            sync_source(self.client, "docs", [chunk], [], "test-model")
        self.assertEqual(self.client.count("docs", exact=True).count, 1)


if __name__ == "__main__":
    unittest.main()
