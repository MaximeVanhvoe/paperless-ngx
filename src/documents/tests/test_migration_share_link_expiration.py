from datetime import timedelta

from django.test import override_settings
from django.utils import timezone

from paperless_testing.migrations import TestMigrations


@override_settings(SHARE_LINK_MAX_DAYS=3)
class TestMigrateShareLinkExpirationBackfill(TestMigrations):
    migrate_from = "0027_documentbarcode"
    migrate_to = "0028_backfill_share_link_expiration"

    def setUpBeforeMigration(self, apps) -> None:
        Document = apps.get_model("documents", "Document")
        ShareLink = apps.get_model("documents", "ShareLink")
        ShareLinkBundle = apps.get_model("documents", "ShareLinkBundle")
        document = Document.objects.create(
            title="doc",
            mime_type="application/pdf",
            checksum="a" * 64,
        )
        self.existing_expiration = timezone.now() + timedelta(days=1)
        self.link_never_id = ShareLink.objects.create(
            document=document,
            slug="never",
        ).pk
        self.link_dated_id = ShareLink.objects.create(
            document=document,
            slug="dated",
            expiration=self.existing_expiration,
        ).pk
        self.bundle_never_id = ShareLinkBundle.objects.create(slug="bundle").pk

    def test_null_expirations_are_backfilled_with_max_days(self) -> None:
        ShareLink = self.apps.get_model("documents", "ShareLink")
        ShareLinkBundle = self.apps.get_model("documents", "ShareLinkBundle")
        expected = timezone.now() + timedelta(days=3)

        for expiration in (
            ShareLink.objects.get(pk=self.link_never_id).expiration,
            ShareLinkBundle.objects.get(pk=self.bundle_never_id).expiration,
        ):
            self.assertIsNotNone(expiration)
            self.assertLess(abs(expiration - expected), timedelta(minutes=5))

        self.assertEqual(
            ShareLink.objects.get(pk=self.link_dated_id).expiration,
            self.existing_expiration,
        )
