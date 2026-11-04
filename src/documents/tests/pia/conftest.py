"""Gedeelde fixtures voor de PIA-tests van groep 2 (Privacy and Big Data, H00Y3a).

Use case: kleine huisartsenpraktijk met twee artsen, een assistent en een
praktijkmanager. Alle documentinhoud is fictief.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from paperless_testing.factories import DocumentFactory
from paperless_testing.factories import UserFactory
from paperless_testing.permissions import grant_global

if TYPE_CHECKING:
    from django.contrib.auth.models import User
    from rest_framework.test import APIClient

    from documents.models import Document

# Globale (model-level) rechten die elke medewerker nodig heeft om de
# documenten-API te mogen gebruiken. Zonder deze rechten geeft de API altijd
# 403 en zou een test om de verkeerde reden slagen.
STAFF_GLOBAL_PERMS = ("view_document", "change_document", "add_document")


def _staff(username: str) -> User:
    user = UserFactory(username=username)
    grant_global(user, *STAFF_GLOBAL_PERMS)
    return user


# --- Gebruikers -----------------------------------------------------------


@pytest.fixture
def gp_a(db: None) -> User:
    """Huisarts A."""
    return _staff("dr_peeters")


@pytest.fixture
def gp_b(db: None) -> User:
    """Huisarts B."""
    return _staff("dr_janssens")


@pytest.fixture
def assistant(db: None) -> User:
    """Praktijkassistent: administratieve rol, geen behandelrelatie."""
    return _staff("assistent")


@pytest.fixture
def practice_manager(db: None) -> User:
    """Praktijkmanager die ook het IT-beheer doet (superuser)."""
    return UserFactory(username="praktijkmanager", is_staff=True, is_superuser=True)


# --- API-clients ----------------------------------------------------------


def _client_for(user: User) -> APIClient:
    from rest_framework.test import APIClient

    client = APIClient()
    client.force_authenticate(user=user)
    client.credentials(HTTP_ACCEPT="application/json; version=10")
    return client


@pytest.fixture
def client_for():
    """Factory: client_for(user) geeft een ingelogde API-client voor die user."""
    return _client_for


# --- Fictieve documenten ----------------------------------------------------


@pytest.fixture
def discharge_letter_gp_a(gp_a: User) -> Document:
    """Ontslagbrief die correct aan huisarts A toegewezen is."""
    return DocumentFactory(
        title="Ontslagbrief AZ - patiënt Test Persoon",
        content=(
            "Ontslagbrief. Patiënt: Test Persoon, RRN 00.00.00-000.00 (ongeldig). "
            "Diagnose: fictieve diagnose voor testdoeleinden."
        ),
        owner=gp_a,
    )


@pytest.fixture
def unowned_lab_result(db: None) -> Document:
    """Labo-uitslag zonder owner, zoals een document uit de consume-map
    of een mailregel zonder ingestelde eigenaar."""
    return DocumentFactory(
        title="Labo-uitslag - patiënt Test Persoon",
        content="Labo-uitslag. Patiënt: Test Persoon. CRP 45 mg/L (fictief).",
        owner=None,
    )
