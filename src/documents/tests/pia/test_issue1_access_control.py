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

import pytest

from documents.tests.pia.helpers import visible_document_ids

pytestmark = [pytest.mark.pia, pytest.mark.django_db]


class TestIssue1OwnerlessDocuments:
    # Controle: het permissiemodel werkt wel voor documenten met owner.
    # Deze test moet slagen; zo tonen we dat het probleem specifiek bij
    # documenten zonder owner zit en niet in onze testopzet.
    def test_assistant_cannot_see_document_owned_by_gp(
        self, assistant, gp_a, discharge_letter_gp_a, client_for
    ):
        assert discharge_letter_gp_a.pk in visible_document_ids(client_for(gp_a))
        assert discharge_letter_gp_a.pk not in visible_document_ids(
            client_for(assistant)
        )

    @pytest.mark.xfail(
        reason="PIA issue 1: documenten zonder owner zijn zichtbaar voor elke gebruiker",
    )
    def test_assistant_cannot_list_unowned_medical_document(
        self, assistant, unowned_lab_result, client_for
    ):
        assert unowned_lab_result.pk not in visible_document_ids(client_for(assistant))

    @pytest.mark.xfail(
        reason="PIA issue 1: documenten zonder owner zijn zichtbaar voor elke gebruiker",
    )
    def test_assistant_cannot_open_unowned_medical_document(
        self, assistant, unowned_lab_result, client_for
    ):
        response = client_for(assistant).get(f"/api/documents/{unowned_lab_result.pk}/")
        assert response.status_code in (403, 404)
