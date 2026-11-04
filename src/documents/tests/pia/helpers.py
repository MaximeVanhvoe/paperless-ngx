"""Hulpfuncties voor de PIA-tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rest_framework.test import APIClient


def visible_document_ids(client: APIClient) -> set[int]:
    """Alle document-id's die deze client via de lijst-API kan zien."""
    response = client.get("/api/documents/", {"page_size": 100000})
    assert response.status_code == 200, response.content
    return {doc["id"] for doc in response.json()["results"]}
