from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import CommandError
from django.db import transaction
from django.db.models import Q

from documents.management.commands.base import PaperlessCommand
from documents.models import Document


class Command(PaperlessCommand):
    """Assign an owner to every existing document that has none."""

    help = (
        "Assigns an owner to every document without an owner, including "
        "documents in the trash. Uses PAPERLESS_CONSUMER_DEFAULT_OWNER unless "
        "--owner is given."
    )

    supports_progress_bar = False
    supports_multiprocessing = False

    def add_arguments(self, parser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--owner",
            default=None,
            help=(
                "Username of the owner to assign. Defaults to "
                "PAPERLESS_CONSUMER_DEFAULT_OWNER."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Only report which documents would get an owner.",
        )

    def handle(self, *args, **options) -> None:
        username = options["owner"] or settings.CONSUMER_DEFAULT_OWNER
        if not username:
            raise CommandError(
                "No owner given: pass --owner or set PAPERLESS_CONSUMER_DEFAULT_OWNER.",
            )
        owner = User.objects.filter(username=username).first()
        if owner is None:
            raise CommandError(f"No user with username '{username}' exists.")

        # Root documents without an owner, and their versions without an
        # owner. A version is authorized by its root, so a version of an owned
        # root keeps its own (unused) owner field as it is.
        unowned = Document.global_objects.filter(owner__isnull=True).filter(
            Q(root_document__isnull=True) | Q(root_document__owner__isnull=True),
        )
        document_ids = list(unowned.order_by("pk").values_list("pk", flat=True))

        if not document_ids:
            self.console.print("No documents without an owner found.")
            return

        if options["dry_run"]:
            self.console.print(
                f"Dry run: {len(document_ids)} document(s) without an owner "
                f"would be assigned to '{owner.username}':",
            )
            for document_id in document_ids:
                self.console.print(f"  document #{document_id}")
            return

        with transaction.atomic():
            updated = Document.global_objects.filter(
                pk__in=document_ids,
                owner__isnull=True,
            ).update(owner=owner)

        # QuerySet.update() sends no signals, so refresh the owner stored in
        # the search index here; it is used to filter search results.
        from documents.search import get_backend

        with get_backend().batch_update() as batch:
            batch.add_or_update_ids(document_ids)

        self.console.print(
            f"Assigned owner '{owner.username}' to {updated} document(s).",
            style="green",
        )
