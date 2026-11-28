"""PIA issue 1 - Toegangscontrole: documenten zonder owner.

Privacy-eis: een medisch document is enkel zichtbaar voor wie er een
behandelrelatie mee heeft. Een administratieve gebruiker (de assistent) mag
geen document zien dat niet aan hem toegewezen of met hem gedeeld is.

Bevinding: Paperless maakt documenten zonder owner zichtbaar voor elke
gebruiker met de globale view-permissie (zie documents/permissions.py,
get_objects_for_user_owner_aware: ``owner__isnull=True``).
LINDDUN: Data disclosure, Non-compliance. AVG: art. 5(1)(f), art. 9, art. 25.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from documents import tasks
from documents.data_models import ConsumableDocument
from documents.data_models import DocumentSource
from documents.models import Document
from documents.models import Workflow
from documents.models import WorkflowAction
from documents.models import WorkflowTrigger
from documents.search import get_backend
from documents.tests.pia.helpers import visible_document_ids
from paperless_mail.mail import MailAccountHandler
from paperless_mail.models import MailAccount
from paperless_mail.models import MailRule
from paperless_mail.tests.helpers import MessageBuilder
from paperless_mail.tests.helpers import _AttachmentDef
from paperless_testing.factories import CorrespondentFactory
from paperless_testing.factories import TagFactory
from paperless_testing.permissions import grant_global
from paperless_testing.permissions import grant_object

if TYPE_CHECKING:
    from django.contrib.auth.models import User
    from pytest_django.fixtures import SettingsWrapper

pytestmark = [pytest.mark.pia, pytest.mark.django_db]

SAMPLE_PDF = Path(__file__).parents[1] / "samples" / "simple.pdf"


class TestIssue1OwnerlessDocuments:
    # Controle: het permissiemodel werkt wel voor documenten met owner.
    # Deze test moet slagen; zo tonen we dat het probleem specifiek bij
    # documenten zonder owner zit en niet in onze testopzet.
    def test_assistant_cannot_see_document_owned_by_gp(
        self,
        assistant,
        gp_a,
        discharge_letter_gp_a,
        client_for,
    ):
        assert discharge_letter_gp_a.pk in visible_document_ids(client_for(gp_a))
        assert discharge_letter_gp_a.pk not in visible_document_ids(
            client_for(assistant),
        )

    def test_assistant_cannot_list_unowned_medical_document(
        self,
        assistant,
        unowned_lab_result,
        client_for,
    ):
        assert unowned_lab_result.pk not in visible_document_ids(client_for(assistant))

    def test_assistant_cannot_open_unowned_medical_document(
        self,
        assistant,
        unowned_lab_result,
        client_for,
    ):
        response = client_for(assistant).get(f"/api/documents/{unowned_lab_result.pk}/")
        assert response.status_code in (403, 404)


# --- Fixtures voor de fix (PIA issue 1) ---------------------------------------


@pytest.fixture
def default_owner(settings: SettingsWrapper, gp_a: User) -> User:
    """PAPERLESS_CONSUMER_DEFAULT_OWNER wijst naar huisarts A, die de intake
    van de praktijk triageert."""
    settings.CONSUMER_DEFAULT_OWNER = gp_a.username
    return gp_a


@pytest.fixture
def intake_file(paperless_dirs, fake_progress_manager) -> Path:
    """Een gescand document dat binnenkomt via de consume-map."""
    return Path(shutil.copy(SAMPLE_PDF, paperless_dirs.scratch_dir / "scan_lab.pdf"))


def consume(path: Path, source: DocumentSource, overrides=None) -> Document:
    tasks.consume_file(ConsumableDocument(source=source, original_file=path), overrides)
    return Document.objects.get()


@pytest.fixture
def indexed_lab_result(paperless_dirs, unowned_lab_result: Document) -> Document:
    """Labo-uitslag zonder owner, met tag en correspondent, in de zoekindex."""
    unowned_lab_result.tags.add(TagFactory(name="labo", owner=None))
    unowned_lab_result.correspondent = CorrespondentFactory(name="Labo X", owner=None)
    unowned_lab_result.save()
    get_backend().add_or_update(unowned_lab_result)
    return unowned_lab_result


def search_ids(client, query: str) -> set[int]:
    response = client.get("/api/documents/", {"query": query})
    assert response.status_code == 200, response.content
    return {doc["id"] for doc in response.json()["results"]}


def document_count(client, endpoint: str, name: str) -> int:
    response = client.get(f"/api/{endpoint}/", {"page_size": 1000})
    assert response.status_code == 200, response.content
    (obj,) = [o for o in response.json()["results"] if o["name"] == name]
    return obj["document_count"]


# --- Wijziging 1: documenten zonder owner verbergen (Hoepman: Hide) ----------


class TestIssue1UnownedDocumentsHidden:
    def test_object_level_grant_makes_unowned_document_visible(
        self,
        assistant,
        gp_b,
        unowned_lab_result,
        client_for,
    ):
        grant_object(assistant, unowned_lab_result, "view_document")

        assert unowned_lab_result.pk in visible_document_ids(client_for(assistant))
        response = client_for(assistant).get(f"/api/documents/{unowned_lab_result.pk}/")
        assert response.status_code == 200
        # De grant geldt enkel voor de assistent, niet voor huisarts B.
        assert unowned_lab_result.pk not in visible_document_ids(client_for(gp_b))

    def test_assistant_cannot_find_unowned_document_via_search(
        self,
        assistant,
        practice_manager,
        indexed_lab_result,
        client_for,
    ):
        # (d) full-text search, autocomplete, more-like-this en globale zoekbalk
        assert indexed_lab_result.pk in search_ids(client_for(practice_manager), "CRP")
        assert search_ids(client_for(assistant), "CRP") == set()

        autocomplete = "/api/search/autocomplete/"
        assert client_for(practice_manager).get(autocomplete, {"term": "crp"}).json()
        assert client_for(assistant).get(autocomplete, {"term": "crp"}).json() == []

        response = client_for(assistant).get(
            "/api/documents/",
            {"more_like_id": indexed_lab_result.pk},
        )
        if response.status_code == 200:
            assert response.json()["results"] == []
        else:
            assert response.status_code in (400, 403, 404)

        response = client_for(assistant).get("/api/search/", {"query": "Labo"})
        assert response.status_code == 200
        assert response.json()["documents"] == []

    def test_assistant_cannot_infer_unowned_document_from_counts(
        self,
        assistant,
        practice_manager,
        indexed_lab_result,
        client_for,
    ):
        # (d) document_count op tags/correspondents en de statistieken
        grant_global(assistant, "view_tag", "view_correspondent")
        assistant_client = client_for(assistant)
        manager_client = client_for(practice_manager)

        assert document_count(manager_client, "tags", "labo") == 1
        assert document_count(assistant_client, "tags", "labo") == 0
        assert document_count(manager_client, "correspondents", "Labo X") == 1
        assert document_count(assistant_client, "correspondents", "Labo X") == 0

        stats = assistant_client.get("/api/statistics/").json()
        assert stats["documents_total"] == 0

    @pytest.mark.parametrize(
        "path",
        ["notes/", "download/", "preview/", "thumb/", "metadata/", "history/"],
    )
    def test_assistant_cannot_reach_unowned_document_endpoints(
        self,
        settings,
        assistant,
        sample_doc,
        client_for,
        path,
    ):
        # sample_doc: document zonder owner met echte bestanden, zodat een 403
        # of 404 niet komt van een ontbrekend bestand.
        settings.AUDIT_LOG_ENABLED = True
        grant_global(assistant, "view_note", "auditlog.view_logentry")
        url = f"/api/documents/{sample_doc.pk}/{path}"
        assert sample_doc.owner is None

        assert client_for(assistant).get(url).status_code in (403, 404)

        # Controle: met een object-level grant lukt het wel.
        grant_object(assistant, sample_doc, "view_document")
        assert client_for(assistant).get(url).status_code == 200

    def test_assistant_cannot_bulk_edit_download_or_share_unowned_document(
        self,
        assistant,
        unowned_lab_result,
        client_for,
    ):
        grant_global(assistant, "add_sharelink", "view_sharelink")
        client = client_for(assistant)
        pk = unowned_lab_result.pk

        response = client.post(
            "/api/documents/bulk_edit/",
            {
                "documents": [pk],
                "method": "modify_tags",
                "parameters": {
                    "add_tags": [],
                    "remove_tags": [],
                },
            },
            format="json",
        )
        assert response.status_code == 403

        response = client.post(
            "/api/documents/bulk_download/",
            {"documents": [pk]},
            format="json",
        )
        assert response.status_code == 403

        response = client.post(
            "/api/share_links/",
            {"document": pk, "file_version": "original"},
            format="json",
        )
        assert response.status_code in (400, 403)

    def test_superuser_still_sees_unowned_document(
        self,
        practice_manager,
        indexed_lab_result,
        client_for,
    ):
        # (e)
        client = client_for(practice_manager)
        assert indexed_lab_result.pk in visible_document_ids(client)
        assert client.get(f"/api/documents/{indexed_lab_result.pk}/").status_code == 200
        assert indexed_lab_result.pk in search_ids(client, "CRP")
        assert document_count(client, "tags", "labo") == 1

    def test_old_behaviour_with_unowned_documents_visible(
        self,
        settings,
        assistant,
        indexed_lab_result,
        client_for,
    ):
        # (f) PAPERLESS_UNOWNED_DOCUMENTS_VISIBLE=true herstelt het oude gedrag.
        settings.UNOWNED_DOCUMENTS_VISIBLE = True
        grant_global(assistant, "view_tag")
        client = client_for(assistant)

        assert indexed_lab_result.pk in visible_document_ids(client)
        assert client.get(f"/api/documents/{indexed_lab_result.pk}/").status_code == 200
        assert indexed_lab_result.pk in search_ids(client, "CRP")
        assert client.get("/api/search/autocomplete/", {"term": "crp"}).json()
        assert document_count(client, "tags", "labo") == 1


# --- Wijziging 2: fallback-owner bij intake -----------------------------------


class TestIssue1DefaultOwnerOnIntake:
    @pytest.mark.parametrize(
        "source",
        [DocumentSource.ConsumeFolder, DocumentSource.ApiUpload],
        ids=["consume-folder", "api-upload"],
    )
    def test_consumed_document_without_owner_gets_default_owner(
        self,
        default_owner,
        assistant,
        intake_file,
        client_for,
        source,
    ):
        # (a) consume-map en een API-upload zonder gebruiker
        document = consume(intake_file, source)

        assert document.owner == default_owner
        assert document.pk not in visible_document_ids(client_for(assistant))

    def test_explicit_owner_is_kept(self, default_owner, gp_b, intake_file):
        from documents.data_models import DocumentMetadataOverrides

        document = consume(
            intake_file,
            DocumentSource.ApiUpload,
            DocumentMetadataOverrides(owner_id=gp_b.pk),
        )

        assert document.owner == gp_b

    def test_unknown_default_owner_logs_warning(
        self,
        settings,
        intake_file,
        caplog,
    ):
        settings.CONSUMER_DEFAULT_OWNER = "bestaat_niet"

        with caplog.at_level(logging.WARNING):
            document = consume(intake_file, DocumentSource.ConsumeFolder)

        assert document.owner is None
        assert "PAPERLESS_CONSUMER_DEFAULT_OWNER" in caplog.text
        assert "bestaat_niet" in caplog.text

    def test_mail_rule_without_owner_assignment_gets_default_owner(
        self,
        default_owner,
        gp_b,
        paperless_dirs,
        fake_progress_manager,
        monkeypatch,
    ):
        # (b) mailregel met assign_owner_from_rule=False
        account = MailAccount.objects.create(name="labo", username="u", password="p")
        rule = MailRule.objects.create(
            name="labo-uitslagen",
            account=account,
            owner=gp_b,
            assign_owner_from_rule=False,
            assign_title_from=MailRule.TitleSource.FROM_FILENAME,
        )
        message = MessageBuilder().create_message(
            attachments=[
                _AttachmentDef(filename="labo.pdf", content=SAMPLE_PDF.read_bytes()),
            ],
        )
        queued = []
        monkeypatch.setattr(
            "paperless_mail.mail.queue_consumption_tasks",
            lambda **kwargs: queued.extend(kwargs["consume_tasks"]),
        )

        assert MailAccountHandler()._handle_message(message, rule) == 1
        (consume_task,) = queued
        assert consume_task.kwargs["overrides"].owner_id is None
        tasks.consume_file(**consume_task.kwargs)

        assert Document.objects.get().owner == default_owner

    def _consumption_workflow(self, **action_fields) -> WorkflowAction:
        trigger = WorkflowTrigger.objects.create(
            type=WorkflowTrigger.WorkflowTriggerType.CONSUMPTION,
            sources=f"{DocumentSource.ConsumeFolder}",
            filter_filename="*scan_lab*",
        )
        action = WorkflowAction.objects.create(**action_fields)
        workflow = Workflow.objects.create(name="intake", order=0)
        workflow.triggers.add(trigger)
        workflow.actions.add(action)
        return action

    def test_workflow_owner_wins_over_default_owner(
        self,
        default_owner,
        gp_b,
        assistant,
        intake_file,
        client_for,
    ):
        # (c) een workflow met assign_owner gaat voor de fallback
        action = self._consumption_workflow(assign_owner=gp_b)
        action.assign_view_users.add(assistant)

        document = consume(intake_file, DocumentSource.ConsumeFolder)

        assert document.owner == gp_b
        assert document.pk in visible_document_ids(client_for(assistant))

    def test_workflow_view_permissions_still_apply_with_default_owner(
        self,
        default_owner,
        assistant,
        gp_b,
        intake_file,
        client_for,
    ):
        # Een workflow die enkel view-rechten geeft (bv. de assistent voor de
        # intake) blijft werken naast de fallback-owner.
        action = self._consumption_workflow()
        action.assign_view_users.add(assistant)

        document = consume(intake_file, DocumentSource.ConsumeFolder)

        assert document.owner == default_owner
        assert document.pk in visible_document_ids(client_for(assistant))
        assert document.pk not in visible_document_ids(client_for(gp_b))
