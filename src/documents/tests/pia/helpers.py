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
    """Traint de classifier op alle documenten en schrijft het modelbestand,
    zoals de uurlijkse taak train_classifier doet."""
    classifier = DocumentClassifier()
    classifier.train()
    classifier.save()
    return classifier


def stored_token(text: str) -> str:
    """Het token waaronder `text` in de vocabulary terechtkomt (na lowercase,
    stopwoorden en stemming), zodat tests niet afhangen van de stemmer."""
    tokens = DocumentClassifier().preprocess_content(text).split()
    assert len(tokens) == 1, f"verwacht één token voor {text!r}, kreeg {tokens}"
    return tokens[0]


def vocabulary_from_file() -> set[str]:
    """Leest de vocabulary uit settings.MODEL_FILE op de normale manier,
    zoals Paperless zelf (handtekening gecontroleerd met SECRET_KEY)."""
    classifier = DocumentClassifier()
    classifier.load()
    return set(classifier.data_vectorizer.vocabulary_)


def vocabulary_without_key(model_file: Path) -> set[str]:
    """Leest de vocabulary zoals een aanvaller met enkel het bestand (of een
    back-up) dat kan: de 32 bytes HMAC overslaan en de rest unpicklen,
    zonder SECRET_KEY."""
    raw = model_file.read_bytes()
    payload = pickle.loads(raw[DocumentClassifier.HMAC_SIZE :])
    data_vectorizer = payload[3]
    return set(data_vectorizer.vocabulary_)
