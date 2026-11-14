"""PIA issue 4 - LLM-suggesties voor een document zonder owner zien alle documenten.

Privacy-eis: de context die naar de externe LLM gaat (inhoud van "vergelijkbare
documenten") bevat enkel documenten die bij de betrokken gebruiker horen.

Bevinding: een workflow vraagt suggesties met ``document.owner`` als gebruiker
(documents/workflows/ai.py). Is die ``None``, dan telt dat in
``get_taxonomy_context`` als "geen beperking" en wordt de vector search niet
gefilterd: tekst van documenten van andere artsen kan in de prompt belanden.
LINDDUN: Data disclosure, Unawareness. AVG: art. 5(1)(c), art. 28, hoofdstuk V.
"""

from __future__ import annotations

from unittest import mock

import pytest

from paperless_ai.ai_classifier import get_taxonomy_context

pytestmark = [pytest.mark.pia, pytest.mark.django_db]

RETRIEVE = "paperless_ai.ai_classifier.retrieve_similar_nodes"


@pytest.fixture
def vector_search_enabled(settings):
    settings.LLM_EMBEDDING_BACKEND = "huggingface"


def _allowed_ids(mocked_retrieve) -> list[int] | None:
    return mocked_retrieve.call_args.kwargs["document_ids"]


class TestIssue4LlmContext:
    # Controle: bij een document met owner wordt de zoekopdracht beperkt tot
    # wat die gebruiker mag zien (het GP-document zit erin, dat van de
    # andere arts niet).
    def test_owned_document_search_is_restricted(
        self, gp_a, gp_b, discharge_letter_gp_a, vector_search_enabled
    ):
        from paperless_testing.factories import DocumentFactory

        other_patient = DocumentFactory(
            title="Brief patiënt van dokter B", content="fictief", owner=gp_b
        )
        with mock.patch(RETRIEVE, return_value=[]) as retrieve:
            get_taxonomy_context(discharge_letter_gp_a, user=gp_a)
        allowed = _allowed_ids(retrieve)
        assert allowed is not None
        assert discharge_letter_gp_a.pk in allowed
        assert other_patient.pk not in allowed

    def test_unowned_document_does_not_pull_other_gps_documents(
        self, gp_a, unowned_lab_result, discharge_letter_gp_a, vector_search_enabled
    ):
        # Zo roept documents/workflows/ai.py het aan: user = document.owner = None.
        assert unowned_lab_result.owner is None
        with mock.patch(RETRIEVE, return_value=[]) as retrieve:
            get_taxonomy_context(unowned_lab_result, user=unowned_lab_result.owner)
        allowed = _allowed_ids(retrieve)
        assert allowed is not None, (
            "document_ids=None: de vector search is niet beperkt; documenten van "
            "alle artsen kunnen in de prompt naar de externe LLM terechtkomen."
        )
        assert discharge_letter_gp_a.pk not in allowed


# --- End-to-end: wat komt er effectief in de prompt naar de LLM? -------------

SECRET_MARKER = "GEHEIME-DIAGNOSE-VAN-DOKTER-B"


@pytest.fixture
def prompt_sent_to_llm(gp_b, vector_search_enabled):
    """Voert de workflow-aanroep uit met een nagebootste vector store (die de
    document_ids-filter respecteert) en een nagebootste LLM. Geeft de prompt
    terug die de LLM zou ontvangen."""
    from paperless_ai.ai_classifier import get_ai_document_classification
    from paperless_testing.factories import DocumentFactory

    other = DocumentFactory(
        title="Dossier patiënt van dokter B",
        content=f"Fictief dossier. {SECRET_MARKER}",
        owner=gp_b,
    )

    def fake_retrieve(document, top_k=5, document_ids=None):
        if document_ids is not None and other.pk not in set(document_ids):
            return []
        node = mock.Mock(metadata={"document_id": other.pk}, score=0.9)
        return [node]

    def run(document):
        captured = {}

        def fake_llm(self, prompt, **kwargs):
            captured.setdefault("prompt", prompt)
            return {}

        with (
            mock.patch(RETRIEVE, side_effect=fake_retrieve),
            mock.patch("paperless_ai.ai_classifier.AIClient.__init__", return_value=None),
            mock.patch("paperless_ai.ai_classifier.AIClient.run_llm_query", fake_llm),
        ):
            get_ai_document_classification(document, document.owner, None)
        return captured["prompt"]

    return run


class TestIssue4EndToEnd:
    # Controle: bij een document van dokter A bevat de prompt niets van dokter B.
    def test_owned_document_prompt_has_no_other_gps_text(
        self, discharge_letter_gp_a, prompt_sent_to_llm
    ):
        assert SECRET_MARKER not in prompt_sent_to_llm(discharge_letter_gp_a)

    def test_unowned_document_prompt_has_no_other_gps_text(
        self, unowned_lab_result, prompt_sent_to_llm
    ):
        assert SECRET_MARKER not in prompt_sent_to_llm(unowned_lab_result)


class TestIssue4OtherPatientsInPrompt:
    """Kern van issue 4: ook mét owner en mét correcte filter bevat de prompt
    voor patiënt X tekst van brieven van andere patiënten, want de GP mag die
    allemaal zien. Die tekst gaat naar de externe LLM-provider."""

    def test_prompt_for_patient_x_contains_no_text_of_patient_y(
        self, gp_a, vector_search_enabled
    ):
        from paperless_ai.ai_classifier import get_ai_document_classification
        from paperless_testing.factories import DocumentFactory

        letter_x = DocumentFactory(
            title="Brief patiënt X", content="Fictieve brief over patiënt X.", owner=gp_a
        )
        letter_y = DocumentFactory(
            title="Brief patiënt Y",
            content="Fictieve brief over patiënt Y. DIAGNOSE-PATIENT-Y",
            owner=gp_a,
        )

        def fake_retrieve(document, top_k=5, document_ids=None):
            # Dokter A mag beide brieven zien, dus de filter laat Y door.
            assert letter_y.pk in set(document_ids)
            return [mock.Mock(metadata={"document_id": letter_y.pk}, score=0.9)]

        captured = {}

        def fake_llm(self, prompt, **kwargs):
            captured.setdefault("prompt", prompt)
            return {}

        with (
            mock.patch(RETRIEVE, side_effect=fake_retrieve),
            mock.patch("paperless_ai.ai_classifier.AIClient.__init__", return_value=None),
            mock.patch("paperless_ai.ai_classifier.AIClient.run_llm_query", fake_llm),
        ):
            get_ai_document_classification(letter_x, gp_a, None)
        assert "DIAGNOSE-PATIENT-Y" not in captured["prompt"]
