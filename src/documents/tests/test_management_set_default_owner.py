from io import StringIO

import pytest
from django.core.management import CommandError
from django.core.management import call_command

from documents.models import Document
from documents.search import get_backend
from paperless_testing.factories import DocumentFactory
from paperless_testing.factories import UserFactory


def run_command(*args: str) -> str:
    out = StringIO()
    call_command("document_set_default_owner", *args, stdout=out)
    return out.getvalue()


@pytest.mark.django_db
@pytest.mark.usefixtures("paperless_dirs")
class TestDocumentSetDefaultOwner:
    @pytest.fixture
    def intake(self, settings):
        settings.CONSUMER_DEFAULT_OWNER = "intake"
        return UserFactory(username="intake")

    @pytest.fixture
    def documents(self):
        other = UserFactory(username="other")
        unowned_root = DocumentFactory(owner=None, content="labo crp")
        trashed = DocumentFactory(owner=None)
        trashed.delete()
        return {
            "unowned_root": unowned_root,
            "unowned_version": DocumentFactory(
                root_document=unowned_root,
                owner=None,
                content="labo crp",
            ),
            "trashed": trashed,
            "owned": DocumentFactory(owner=other),
            "version_of_owned": DocumentFactory(
                root_document=DocumentFactory(owner=other),
                owner=None,
            ),
            "other": other,
        }

    def test_assigns_default_owner_to_unowned_documents(self, intake, documents):
        """
        GIVEN:
            - PAPERLESS_CONSUMER_DEFAULT_OWNER names an existing user
            - Unowned documents (a root, its version, one in the trash) and
              owned documents
        WHEN:
            - The command runs
        THEN:
            - Every unowned document, trashed included, gets the default owner
            - Owned documents and versions of owned roots are left alone
            - The search index knows the new owner
        """
        get_backend().add_or_update(documents["unowned_root"])

        output = run_command()

        def owner_of(key):
            return Document.global_objects.get(pk=documents[key].pk).owner

        assert owner_of("unowned_root") == intake
        assert owner_of("unowned_version") == intake
        assert owner_of("trashed") == intake
        assert owner_of("owned") == documents["other"]
        assert owner_of("version_of_owned") is None
        assert "3 document(s)" in output
        assert get_backend().search_ids("crp", user=intake) == [
            documents["unowned_root"].pk,
        ]

    def test_dry_run_changes_nothing(self, intake, documents):
        output = run_command("--dry-run")

        assert Document.global_objects.filter(owner=intake).count() == 0
        assert "3 document(s)" in output
        assert f"document #{documents['unowned_root'].pk}" in output

    def test_owner_option_overrides_setting(self, intake, documents):
        manager = UserFactory(username="manager")

        run_command("--owner", "manager")

        assert Document.objects.get(pk=documents["unowned_root"].pk).owner == manager

    def test_without_owner_configured_fails(self, settings, documents):
        settings.CONSUMER_DEFAULT_OWNER = None

        with pytest.raises(CommandError, match="PAPERLESS_CONSUMER_DEFAULT_OWNER"):
            run_command()

    def test_unknown_owner_fails(self, settings, documents):
        settings.CONSUMER_DEFAULT_OWNER = "nobody"

        with pytest.raises(CommandError, match="nobody"):
            run_command()
        assert Document.global_objects.filter(owner__isnull=False).count() == 2

    def test_nothing_to_do(self, intake):
        assert "No documents without an owner" in run_command()
