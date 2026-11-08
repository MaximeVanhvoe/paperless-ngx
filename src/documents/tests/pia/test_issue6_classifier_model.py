"""PIA issue 6 - Classifier: patiëntgegevens in het modelbestand.

Privacy-eis: het classificatiemodel bevat geen gegevens die patiënten
identificeren. Afgeleide data volgt dezelfde toegangsgrenzen als de documenten
zelf, en wat uit Paperless verdwijnt, verdwijnt ook uit het model.

Bevinding: de classifier (documents/classifier.py) traint één CountVectorizer
op de OCR-tekst van alle documenten buiten de inbox, zonder filter op owner of
permissies. Elk woord en woordpaar dat in minstens 1 % van de documenten
voorkomt (min_df=0.01), komt in ``vocabulary_`` en wordt met pickle in
MODEL_FILE opgeslagen. Het bestand is ondertekend met een HMAC (hash functie)
maar niet versleuteld, dit checkt integriteit maar geen vertrouwelijkheid.
Een document in de prullenbak blijft in het model tot de volgende hertraining.
LINDDUN: Data disclosure, Non-compliance.
AVG: art. 5(1)(c), art. 5(1)(f), art. 9, art. 17, art. 25, art. 32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from documents.classifier import ClassifierModelCorruptError
from documents.classifier import DocumentClassifier
from documents.tests.pia.helpers import stored_token
from documents.tests.pia.helpers import train_and_save_classifier
from documents.tests.pia.helpers import vocabulary_from_file
from documents.tests.pia.helpers import vocabulary_without_key
from paperless_testing.factories import DocumentFactory

if TYPE_CHECKING:
    from pathlib import Path

    from django.contrib.auth.models import User

    from documents.models import Document

pytestmark = [pytest.mark.pia, pytest.mark.django_db]

# Fictieve identificatoren. Het rijksregisternummer heeft ongeldige
# controlecijfers. Elk komt in precies één document voor.
SURNAME_PATIENT_GP_A = "Vandenbroeckx"
RRN_PATIENT_GP_A = "85071512345"
SURNAME_PATIENT_GP_B = "Verhaegenbos"


@pytest.fixture
def letter_gp_a(gp_a: User) -> Document:
    """Ontslagbrief van een patiënt van huisarts A, correct aan A toegewezen."""
    return DocumentFactory(
        title="Ontslagbrief cardiologie",
        content=(
            f"Ontslagbrief dienst cardiologie. Patiënt {SURNAME_PATIENT_GP_A}, "
            f"rijksregisternummer {RRN_PATIENT_GP_A}. Opgenomen met "
            "voorkamerfibrillatie, gestart met anticoagulantia."
        ),
        owner=gp_a,
    )


@pytest.fixture
def letter_gp_b(gp_b: User) -> Document:
    """Verslag van een patiënt van huisarts B, correct aan B toegewezen."""
    return DocumentFactory(
        title="Verslag psychiatrie",
        content=(
            f"Verslag consultatie psychiatrie. Patiënt {SURNAME_PATIENT_GP_B}. "
            "Diagnose depressie, opvolging bij de huisarts."
        ),
        owner=gp_b,
    )


class TestIssue6PatientDataInModel:
    @pytest.mark.xfail(
        reason="PIA issue 6: de vocabulary van het model bevat namen en "
        "rijksregisternummers uit documenten (dataminimalisatie)",
    )
    @pytest.mark.parametrize(
        "identifier",
        [SURNAME_PATIENT_GP_A, RRN_PATIENT_GP_A],
        ids=["familienaam", "rijksregisternummer"],
    )
    def test_patient_identifier_not_in_vocabulary(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
        identifier: str,
    ):
        train_and_save_classifier()

        assert stored_token(identifier) not in vocabulary_from_file()

    @pytest.mark.xfail(
        reason="PIA issue 6: het modelbestand is niet versleuteld; een naam "
        "staat leesbaar in de bytes (Confidentiality at rest)",
    )
    def test_patient_identifier_not_in_raw_model_file(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
    ):
        train_and_save_classifier()

        raw = classifier_model_file.read_bytes()
        assert stored_token(SURNAME_PATIENT_GP_A).encode() not in raw

    @pytest.mark.xfail(
        reason="PIA issue 6: de HMAC beschermt enkel de integriteit; zonder "
        "SECRET_KEY is het model gewoon te unpicklen",
    )
    def test_model_not_readable_without_secret_key(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
    ):
        train_and_save_classifier()

        with pytest.raises(Exception):
            vocabulary_without_key(classifier_model_file)

    @pytest.mark.xfail(
        reason="PIA issue 6: één model bevat de inhoud van documenten van "
        "alle owners, zonder rekening te houden met het permissiemodel",
    )
    def test_documents_of_other_owners_not_in_shared_model(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
        letter_gp_b: Document,
    ):
        train_and_save_classifier()

        vocabulary = vocabulary_from_file()
        from_gp_a = stored_token(SURNAME_PATIENT_GP_A) in vocabulary
        from_gp_b = stored_token(SURNAME_PATIENT_GP_B) in vocabulary
        assert not (from_gp_a and from_gp_b)


class TestIssue6Controls:
    # Controle: de HMAC doet wel wat hij belooft. Een gewijzigd modelbestand
    # wordt geweigerd. Het probleem is dus vertrouwelijkheid, niet integriteit.
    def test_tampered_model_file_is_rejected(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
    ):
        train_and_save_classifier()
        raw = bytearray(classifier_model_file.read_bytes())
        raw[DocumentClassifier.HMAC_SIZE] ^= 0xFF
        classifier_model_file.write_bytes(bytes(raw))

        with pytest.raises(ClassifierModelCorruptError):
            DocumentClassifier().load()

    # Controle: min_df=0.01 is de enige filter. Bij 150 documenten valt een
    # term die in 1 document staat weg (1 < 1,5), een term in 2 documenten
    # blijft. Zo tonen we dat onze testopzet de vocabulary correct leest, en
    # dat in een groot archief de namen van frequente patiënten overblijven.
    def test_term_below_min_df_not_stored(self, classifier_model_file: Path):
        DocumentFactory.create_batch(147, content="Algemene brief zonder patiënt.")
        DocumentFactory(content=f"Labo-uitslag patiënt {SURNAME_PATIENT_GP_A}.")
        DocumentFactory.create_batch(
            2,
            content=f"Verslag patiënt {SURNAME_PATIENT_GP_B}.",
        )

        train_and_save_classifier()

        vocabulary = vocabulary_from_file()
        assert stored_token(SURNAME_PATIENT_GP_A) not in vocabulary
        assert stored_token(SURNAME_PATIENT_GP_B) in vocabulary


class TestIssue6Erasure:
    @pytest.mark.xfail(
        reason="PIA issue 6: een document in de prullenbak blijft in het model "
        "tot de volgende hertraining (recht op wissing), ook gelinked aan issue 7",
    )
    def test_trashed_document_removed_from_model(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
        letter_gp_b: Document,
    ):
        train_and_save_classifier()

        letter_gp_a.delete()  # soft delete: naar de prullenbak

        assert stored_token(SURNAME_PATIENT_GP_A) not in vocabulary_from_file()
