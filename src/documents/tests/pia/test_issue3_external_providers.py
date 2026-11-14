"""PIA issue 3 (OCR-deel) - Scans gaan standaard naar een externe OCR-dienst.

Privacy-eis: documenten met gezondheidsgegevens gaan enkel naar een derde partij
als dat per document bewust gekozen is, en over een versleutelde verbinding.

Bevinding: zodra Azure Document Intelligence geconfigureerd is, staat
``remote_ocr_mode`` op ``always``: elke scan gaat naar Azure. De endpoint-validatie
laat ook ``http://`` toe. LINDDUN: Data disclosure, Unawareness. AVG: art. 9, 28,
32, hoofdstuk V.
"""

from __future__ import annotations

import pytest
from django.test import override_settings

from paperless.config import RemoteOCRConfig

pytestmark = [pytest.mark.pia, pytest.mark.django_db]

AZURE = {
    "REMOTE_OCR_ENGINE": "azureai",
    "REMOTE_OCR_API_KEY": "fictieve-sleutel",
    "REMOTE_OCR_ENDPOINT": "https://praktijk-demo.cognitiveservices.azure.com",
}


class TestIssue3RemoteOcr:
    # Controle: zonder Azure-configuratie gaat er niets naar buiten.
    def test_without_azure_no_remote_engine_is_configured(self):
        assert RemoteOCRConfig().remote_ocr_engine is None

    @pytest.mark.xfail(strict=True, reason="v3.3.1: PAPERLESS_REMOTE_OCR_MODE staat standaard op 'always'")
    @override_settings(**AZURE)
    def test_azure_is_not_used_for_every_document_by_default(self):
        assert RemoteOCRConfig().remote_ocr_by_default is False

    # Controle: een https-endpoint wordt aanvaard (de validatie werkt dus wel).
    def test_https_endpoint_is_accepted(self, client_for, practice_manager):
        response = client_for(practice_manager).patch(
            "/api/config/1/",
            {
                "remote_ocr_engine": "azureai",
                "remote_ocr_endpoint": "https://ocr.example.org",
            },
            format="json",
        )
        assert response.status_code == 200, response.content

    @pytest.mark.xfail(strict=True, reason="v3.3.1: validate_outbound_http_url laat http:// toe")
    def test_plain_http_endpoint_is_rejected(self, client_for, practice_manager):
        response = client_for(practice_manager).patch(
            "/api/config/1/",
            {
                "remote_ocr_engine": "azureai",
                "remote_ocr_endpoint": "http://ocr.example.org",
            },
            format="json",
        )
        assert response.status_code == 400


# --- LLM-deel van issue 3: documenttekst naar een externe LLM-provider -------


class TestIssue3ExternalLlm:
    # Controle: een https-endpoint voor de LLM wordt aanvaard.
    def test_https_llm_endpoint_is_accepted(self, client_for, practice_manager):
        from unittest import mock

        with mock.patch("paperless.serialisers.validate_outbound_http_url"):
            response = client_for(practice_manager).patch(
                "/api/config/1/",
                {"llm_backend": "openai-like", "llm_endpoint": "https://llm.example.org"},
                format="json",
            )
        assert response.status_code == 200, response.content

    @pytest.mark.xfail(strict=True, reason="v3.3.1: validate_outbound_http_url laat http:// toe")
    def test_plain_http_llm_endpoint_is_rejected(
        self, client_for, practice_manager, settings
    ):
        settings.LLM_ALLOW_INTERNAL_ENDPOINTS = False
        response = client_for(practice_manager).patch(
            "/api/config/1/",
            {"llm_backend": "openai-like", "llm_endpoint": "http://93.184.216.34"},
            format="json",
        )
        assert response.status_code == 400

    @pytest.mark.xfail(strict=True, reason="v3.3.1: geen redactie voor de prompt")
    def test_prompt_does_not_contain_national_number(self, unowned_lab_result):
        from unittest import mock

        from paperless_ai.ai_classifier import get_ai_document_classification

        unowned_lab_result.content = (
            "Patient Test Persoon, rijksregisternummer 00.00.00-000.00, CRP 45."
        )
        unowned_lab_result.save()
        captured = {}

        def fake_llm(self, prompt, **kwargs):
            captured.setdefault("prompt", prompt)
            return {}

        with (
            mock.patch(
                "paperless_ai.ai_classifier.get_taxonomy_context",
                return_value=({"tags": [], "document_types": [], "correspondents": [], "storage_paths": []}, ""),
            ),
            mock.patch("paperless_ai.ai_classifier.AIClient.__init__", return_value=None),
            mock.patch("paperless_ai.ai_classifier.AIClient.run_llm_query", fake_llm),
        ):
            get_ai_document_classification(unowned_lab_result, None, None)
        assert "00.00.00-000.00" not in captured["prompt"]
