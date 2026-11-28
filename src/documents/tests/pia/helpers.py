"""Hulpfuncties voor de PIA-tests."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from documents.classifier import DocumentClassifier

if TYPE_CHECKING:
    from pathlib import Path

    from rest_framework.test import APIClient


def visible_document_ids(client: APIClient) -> set[int]:
    """Alle document-id's die deze client via de lijst-API kan zien."""
    response = client.get("/api/documents/", {"page_size": 100000})
    assert response.status_code == 200, response.content
    return {doc["id"] for doc in response.json()["results"]}


# --- Classifier -------------------------------------------------------------


def train_and_save_classifier() -> DocumentClassifier:
    """Train the classifier on all documents and write the model file,
    as the hourly train_classifier task does."""
    classifier = DocumentClassifier()
    classifier.train()
    classifier.save()
    return classifier


def stored_token(text: str) -> str:
    """Find the token under which the string `text` ends up in the vocabulary
    (after lowercasing, stop words and stemming), so tests do not depend on
    the stemmer."""
    tokens = DocumentClassifier().preprocess_content(text).split()
    assert len(tokens) == 1, f"expected one token for {text!r}, got {tokens}"
    return tokens[0]


def vocabulary_from_file() -> set[str]:
    """Read the stored terms from settings.MODEL_FILE the normal way, as
    Paperless itself does (signature checked with SECRET_KEY).

    A CountVectorizer stores its terms in ``vocabulary_``. The hashed vectorizer
    has no such attribute because it stores no terms, so the result is empty.
    """
    classifier = DocumentClassifier()
    classifier.load()
    return set(getattr(classifier.data_vectorizer, "vocabulary_", {}))


def unpickle_without_key(model_file: Path) -> object:
    """Read the model as an attacker with only the file (or a backup) can:
    skip the 32-byte HMAC and unpickle the rest, without SECRET_KEY."""
    raw = model_file.read_bytes()
    return pickle.loads(raw[DocumentClassifier.HMAC_SIZE :])
