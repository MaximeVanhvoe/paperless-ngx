from datetime import timedelta

from django.conf import settings
from django.db import migrations
from django.utils import timezone


def backfill_share_link_expiration(apps, schema_editor):
    """
    Give share links and bundles that never expire an expiration of
    now + PAPERLESS_SHARE_LINK_MAX_DAYS, so no existing link stays valid forever.
    """
    expiration = timezone.now() + timedelta(days=settings.SHARE_LINK_MAX_DAYS)
    for model_name in ("ShareLink", "ShareLinkBundle"):
        model = apps.get_model("documents", model_name)
        model.objects.filter(expiration__isnull=True).update(expiration=expiration)


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0027_documentbarcode"),
    ]

    operations = [
        migrations.RunPython(
            backfill_share_link_expiration,
            migrations.RunPython.noop,
        ),
    ]
