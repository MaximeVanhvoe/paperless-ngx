"""PIA issue 8 - Share links: no login, no expiry cap, no audit trail.

Privacy requirement: a share link to a medical document must be bounded in
time, have a minimal perimeter (only the document owner may share), and every
external access must leave an audit trail - the practice carries the
accountability (GDPR Art. 5(2)).

Findings: paperless-ngx accepts share links without an expiration date, has no
upper bound on the expiration, grants creation on view-only object access
(contradicting docs/usage.md which states only owners may share), produces no
log of anonymous accesses via /share/<slug>/, keeps expired single-link rows
forever (no cleanup task equivalent to the bundle one), and re-activates an
existing share link when the document is restored from the trash.

LINDDUN: Data Disclosure, Unawareness & Unintervenability, Non-compliance.
GDPR: Art. 5(1)(e), 5(1)(f), 5(2), 9, 25(2), 32.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest
from django.http import HttpResponse
from django.test import Client
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework.test import APIClient

from documents.models import Document
from documents.models import ShareLink
from paperless_testing.factories import DocumentFactory
from paperless_testing.factories import UserFactory
from paperless_testing.permissions import grant_global
from paperless_testing.permissions import grant_object

pytestmark = [pytest.mark.pia, pytest.mark.django_db]


# --- Issue-8-specific fixtures (kept out of pia/conftest.py on purpose) ---

SHARELINK_GLOBAL_PERMS = (
    "view_sharelink",
    "add_sharelink",
    "delete_sharelink",
)


def _grant_sharelink(user) -> None:
    grant_global(user, *SHARELINK_GLOBAL_PERMS)


@pytest.fixture
def discharge_letter_jan_peeters(gp_a) -> Document:
    """Fictional discharge letter owned by a GP. Patient and content invented."""
    _grant_sharelink(gp_a)
    return DocumentFactory(
        title="Discharge letter - patient Jan Peeters",
        content=(
            "Discharge letter. Patient: Jan Peeters, national-id "
            "00.00.00-000.00 (invalid). Diagnosis: fictional for test."
        ),
        owner=gp_a,
        filename="jan-peeters-discharge.pdf",
        mime_type="application/pdf",
    )


@pytest.fixture
def shared_viewer(discharge_letter_jan_peeters):
    """A colleague with only object-level view on the letter.

    Not the owner, no change_document permission. Still gets the global
    add_sharelink and view_document permissions because the privacy question
    is whether object-level view alone opens the door to sharing.
    """
    user = UserFactory(username="dr_dupont")
    grant_global(user, "view_document", "add_document")
    _grant_sharelink(user)
    grant_object(user, discharge_letter_jan_peeters, "view_document")
    return user


def _anonymous_django_client() -> Client:
    # The share endpoint is a plain Django View, not a DRF view. Use
    # raise_request_exception=False so a crash inside the view surfaces as a
    # 500 instead of propagating to the test.
    return Client(raise_request_exception=False)


# --- Tests ---------------------------------------------------------------


class TestIssue8ShareLinks:
    # ---- Control tests (should pass today) ---------------------------

    def test_expired_link_is_denied(self, discharge_letter_jan_peeters) -> None:
        """An expired share link must not serve the file."""
        link = ShareLink.objects.create(
            document=discharge_letter_jan_peeters,
            slug="expiredslugaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            expiration=timezone.now() - timedelta(days=1),
        )
        response = _anonymous_django_client().get(f"/share/{link.slug}/")
        assert response.status_code == 302
        assert response["Location"].endswith("sharelink_expired=1")

    def test_unknown_slug_is_rejected(self) -> None:
        """An unknown slug must redirect to the login page with a not-found marker."""
        response = _anonymous_django_client().get(
            "/share/doesnotexistaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/",
        )
        assert response.status_code == 302
        assert response["Location"].endswith("sharelink_notfound=1")

    def test_anonymous_cannot_list_share_links_via_api(
        self,
        discharge_letter_jan_peeters,
    ) -> None:
        """An unauthenticated caller must not enumerate share links through the API."""
        ShareLink.objects.create(
            document=discharge_letter_jan_peeters,
            slug="listslugaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        anon = APIClient()
        anon.credentials(HTTP_ACCEPT="application/json; version=10")
        response = anon.get("/api/share_links/")
        assert response.status_code in (401, 403)

    # ---- Privacy xfails ---------------------------------------------

    @pytest.mark.xfail(
        reason="PIA issue 8: a share link without expiration is accepted (no cap).",
    )
    def test_link_without_expiration_is_rejected_or_capped(
        self,
        gp_a,
        discharge_letter_jan_peeters,
        client_for,
    ) -> None:
        """Creating a link without an expiration date must fail or be capped."""
        response = client_for(gp_a).post(
            "/api/share_links/",
            data={"document": discharge_letter_jan_peeters.pk},
            format="json",
        )
        if response.status_code == 400:
            return
        assert response.status_code == 201
        expiration_raw = response.data["expiration"]
        assert expiration_raw is not None
        max_cap = timezone.now() + timedelta(days=7, hours=1)
        assert parse_datetime(expiration_raw) <= max_cap

    @pytest.mark.xfail(
        reason="PIA issue 8: an expiration far in the future is not capped.",
    )
    def test_expiration_beyond_maximum_is_rejected_or_capped(
        self,
        gp_a,
        discharge_letter_jan_peeters,
        client_for,
    ) -> None:
        """An expiration beyond a sensible maximum must be rejected or capped."""
        far_future = (timezone.now() + timedelta(days=365)).isoformat()
        response = client_for(gp_a).post(
            "/api/share_links/",
            data={
                "document": discharge_letter_jan_peeters.pk,
                "expiration": far_future,
            },
            format="json",
        )
        if response.status_code == 400:
            return
        assert response.status_code == 201
        max_cap = timezone.now() + timedelta(days=7, hours=1)
        assert parse_datetime(response.data["expiration"]) <= max_cap

    @pytest.mark.xfail(
        reason=(
            "PIA issue 8: a view-only colleague can create a share link, "
            "contradicting docs/usage.md which says only owners may share."
        ),
    )
    def test_object_viewer_without_change_cannot_create_link(
        self,
        shared_viewer,
        discharge_letter_jan_peeters,
        client_for,
    ) -> None:
        """Object-level view (without change, not owner) must not allow link creation."""
        response = client_for(shared_viewer).post(
            "/api/share_links/",
            data={"document": discharge_letter_jan_peeters.pk},
            format="json",
        )
        assert response.status_code == 403

    # ---- Trash-related behaviour ------------------------------------

    def test_link_to_trashed_document_is_not_served(
        self,
        discharge_letter_jan_peeters,
        mocker,
    ) -> None:
        """A link whose document is in the trash must not serve the file.

        serve_file is mocked so that reaching it is evidence the view handed
        the trashed file to the caller; the test only passes when the view
        short-circuits before that call.
        """
        served = mocker.patch(
            "documents.views.serve_file",
            return_value=HttpResponse(
                b"fictional pdf bytes",
                content_type="application/pdf",
                status=200,
            ),
        )
        link = ShareLink.objects.create(
            document=discharge_letter_jan_peeters,
            slug="trashedslugaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            expiration=timezone.now() + timedelta(days=1),
            file_version=ShareLink.FileVersion.ORIGINAL,
        )
        discharge_letter_jan_peeters.delete()  # soft delete; document is in the trash
        response = _anonymous_django_client().get(f"/share/{link.slug}/")
        print(
            f"[issue8] trashed-doc status={response.status_code} "
            f"location={response.get('Location')!r} serve_file_called={served.called}",
        )
        assert response.status_code != 200
        assert not served.called

    @pytest.mark.xfail(
        reason=(
            "PIA issue 8: share link silently becomes valid again after the "
            "document is restored from the trash."
        ),
    )
    def test_link_is_revoked_after_document_is_restored(
        self,
        gp_a,
        discharge_letter_jan_peeters,
        client_for,
        mocker,
    ) -> None:
        """A link issued before the document was trashed must stay revoked after restore."""
        served = mocker.patch(
            "documents.views.serve_file",
            return_value=HttpResponse(
                b"fictional pdf bytes",
                content_type="application/pdf",
                status=200,
            ),
        )
        link = ShareLink.objects.create(
            document=discharge_letter_jan_peeters,
            slug="restoredslugaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            expiration=timezone.now() + timedelta(days=1),
            file_version=ShareLink.FileVersion.ORIGINAL,
        )
        # Mirror the real user flow: trash the document, then restore it
        # through the trash API as the owner. The trash endpoint requires
        # global delete_document; grant it to gp_a here.
        grant_global(gp_a, "delete_document")
        discharge_letter_jan_peeters.delete()
        trash_response = client_for(gp_a).post(
            "/api/trash/",
            data={
                "action": "restore",
                "documents": [discharge_letter_jan_peeters.pk],
            },
            format="json",
        )
        assert trash_response.status_code == 200, trash_response.content
        response = _anonymous_django_client().get(f"/share/{link.slug}/")
        print(
            f"[issue8] restored-doc trash_status={trash_response.status_code} "
            f"share_status={response.status_code} "
            f"location={response.get('Location')!r} "
            f"serve_file_called={served.called}",
        )
        assert response.status_code != 200

    # ---- Accountability --------------------------------------------

    @pytest.mark.xfail(
        reason="PIA issue 8: anonymous share-link access leaves no log record.",
    )
    def test_share_link_access_is_logged(
        self,
        discharge_letter_jan_peeters,
        caplog,
        mocker,
    ) -> None:
        """An anonymous visit to /share/<slug>/ must be recorded in the application log.

        The test asserts that some record at INFO or above mentions the slug,
        without coupling to a specific logger name.
        """
        mocker.patch(
            "documents.views.serve_file",
            return_value=HttpResponse(
                b"fictional pdf bytes",
                content_type="application/pdf",
                status=200,
            ),
        )
        link = ShareLink.objects.create(
            document=discharge_letter_jan_peeters,
            slug="loggedslugaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            expiration=timezone.now() + timedelta(days=1),
            file_version=ShareLink.FileVersion.ORIGINAL,
        )
        slug = link.slug
        with caplog.at_level(logging.INFO):
            _anonymous_django_client().get(f"/share/{slug}/")
        assert any(
            slug in record.getMessage() for record in caplog.records
        ), "no log record mentioned the share-link slug"
