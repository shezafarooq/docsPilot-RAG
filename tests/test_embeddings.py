import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.embeddings import get_embedding_model


class TestModelCache(unittest.TestCase):
    def setUp(self):
        get_embedding_model.cache_clear()
        settings = patch("app.embeddings.get_settings", return_value=SimpleNamespace(embedding_model="test-model"))
        settings.start()
        self.addCleanup(settings.stop)
        self.addCleanup(get_embedding_model.cache_clear)

    def test_answering_loads_cached_model_without_network_checks(self):
        with patch("app.embeddings.SentenceTransformer", return_value=object()) as factory:
            first = get_embedding_model()
            second = get_embedding_model()
        self.assertIs(first, second)
        factory.assert_called_once_with("test-model", device="cpu", local_files_only=True)

    def test_uncached_answering_does_not_start_a_model_download(self):
        with patch("app.embeddings.SentenceTransformer", side_effect=OSError("Not cached")) as factory:
            with self.assertRaisesRegex(ValueError, "ingestion"):
                get_embedding_model()
        self.assertEqual(factory.call_count, 1)

    def test_ingestion_can_download_an_uncached_model(self):
        model = object()
        with patch("app.embeddings.SentenceTransformer", side_effect=[OSError("Not cached"), model]) as factory:
            self.assertIs(get_embedding_model(allow_download=True), model)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.kwargs, {"device": "cpu"})
