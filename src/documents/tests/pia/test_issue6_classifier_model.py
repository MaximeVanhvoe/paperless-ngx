"""PIA issue 6 - Classifier: patient data in the model file.

Privacy requirement: the classification model contains no data that
identifies patients, and what disappears from Paperless also disappears from
the model.

Finding: the classifier (documents/classifier.py) trained one CountVectorizer
on the OCR text of all documents outside the inbox, without a filter on owner
or permissions. Every word and word pair that occurs in at least 1 % of the
documents (min_df=0.01) ended up in ``vocabulary_`` and was stored with pickle
in MODEL_FILE. The file is signed with an HMAC (a keyed hash) but not
encrypted: this checks integrity, not confidentiality. A document in the
trash remained in the model until the next hourly retraining.

Fix: hashed features with the same min_df filter store no terms, national
register numbers are removed before vectorizing, and moving a document to the trash schedules a
retraining. The model file is still not encrypted (still xfail).
LINDDUN: Data disclosure, Non-compliance.
GDPR: Art. 5(1)(c), art. 5(1)(f), art. 9, art. 17, art. 25, art. 32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from django.core.cache import cache

from documents import tasks
from documents.classifier import ClassifierModelCorruptError
from documents.classifier import DocumentClassifier
from documents.models import MatchingModel
from documents.signals.handlers import CLASSIFIER_RETRAIN_PENDING_KEY
from documents.tests.pia.helpers import stored_token
from documents.tests.pia.helpers import train_and_save_classifier
from documents.tests.pia.helpers import unpickle_without_key
from documents.tests.pia.helpers import vocabulary_from_file
from paperless_testing.factories import DocumentFactory
from paperless_testing.factories import TagFactory

if TYPE_CHECKING:
    from pathlib import Path

    from django.contrib.auth.models import User

    from documents.models import Document
    from documents.models import Tag

pytestmark = [pytest.mark.pia, pytest.mark.django_db]

# Fictitious identifiers. The national register number has invalid check
# digits. Each occurs in exactly one document.
SURNAME_PATIENT_GP_A = "Vandenbroeckx"
RRN_PATIENT_GP_A = "85071512345"
SURNAME_PATIENT_GP_B = "Verhaegenbos"

# The layouts in which letters write a national register number.
RRN_LAYOUTS = ["85071512345", "85.07.15-123.45", "850715 123 45"]


@pytest.fixture
def letter_gp_a(gp_a: User) -> Document:
    """Discharge letter of a patient of GP A, correctly assigned to A."""
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
    """Report on a patient of GP B, correctly assigned to B."""
    return DocumentFactory(
        title="Verslag psychiatrie",
        content=(
            f"Verslag consultatie psychiatrie. Patiënt {SURNAME_PATIENT_GP_B}. "
            "Diagnose depressie, opvolging bij de huisarts."
        ),
        owner=gp_b,
    )


@pytest.fixture
def auto_tag() -> Tag:
    """A tag with automatic matching. Without one, the train_classifier task
    deletes the model file instead of training it."""
    return TagFactory(name="Klinisch", matching_algorithm=MatchingModel.MATCH_AUTO)


class TestIssue6PatientDataInModel:
    def test_patient_surname_not_in_vocabulary(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
    ):
        train_and_save_classifier()

        assert stored_token(SURNAME_PATIENT_GP_A) not in vocabulary_from_file()

    @pytest.mark.parametrize("rrn", RRN_LAYOUTS, ids=["plain", "dotted", "spaced"])
    def test_national_register_number_never_reaches_the_model(self, rrn: str):
        # Checks the text the vectorizer receives, so the number is absent
        # even as a hash or as fragments such as "85" "07" "15".
        processed = DocumentClassifier().preprocess_content(
            f"Patiënt {SURNAME_PATIENT_GP_A}, rijksregisternummer {rrn}.",
        )

        digits_left = "".join(ch for ch in processed if ch.isdigit())
        assert digits_left == "", processed

    def test_patient_identifier_not_in_raw_model_file(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
    ):
        train_and_save_classifier()

        raw = classifier_model_file.read_bytes()
        assert stored_token(SURNAME_PATIENT_GP_A).encode() not in raw
        assert RRN_PATIENT_GP_A.encode() not in raw

    # min_df keeps rare names out of the features, also with hashing. With
    # 150 documents, a name in 1 document is dropped (1 < 1.5), so it changes
    # no prediction and hashing a guessed name finds no weights. A name in 2
    # documents is kept, which shows that the test can detect a name.
    def test_rare_name_has_no_effect_on_predictions(
        self,
        classifier_model_file: Path,
    ):
        DocumentFactory.create_batch(147, content="Algemene brief zonder patiënt.")
        DocumentFactory(content=f"Verslag patiënt {SURNAME_PATIENT_GP_A}.")
        DocumentFactory.create_batch(
            2,
            content=f"Verslag patiënt {SURNAME_PATIENT_GP_B}.",
        )
        train_and_save_classifier()
        classifier = DocumentClassifier()
        classifier.load()

        def features(text: str):
            processed = classifier.preprocess_content(text)
            return classifier.data_vectorizer.transform([processed])

        without_name = features("Verslag patiënt.")
        with_rare_name = features(f"Verslag patiënt {SURNAME_PATIENT_GP_A}.")
        with_frequent_name = features(f"Verslag patiënt {SURNAME_PATIENT_GP_B}.")
        assert (with_rare_name != without_name).nnz == 0
        assert (with_frequent_name != without_name).nnz > 0

    @pytest.mark.xfail(
        reason="PIA issue 6: the HMAC only protects integrity; without "
        "SECRET_KEY the model can simply be unpickled",
    )
    def test_model_not_readable_without_secret_key(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
    ):
        train_and_save_classifier()

        with pytest.raises(Exception):
            unpickle_without_key(classifier_model_file)


class TestIssue6Controls:
    # Control: the HMAC does what it promises. A modified model file is
    # rejected, so the problem is confidentiality, not integrity.
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

    # Control: the classifier keeps its purpose after the fix. Trained on
    # cardiology letters with an automatic tag, it still suggests that tag
    # for a new cardiology letter and not for an invoice.
    def test_classifier_still_suggests_tags(
        self,
        classifier_model_file: Path,
        auto_tag: Tag,
    ):
        for i in range(8):
            DocumentFactory(
                content=(
                    f"Ontslagbrief cardiologie patiënt nummer {i}. "
                    "Voorkamerfibrillatie, gestart met anticoagulantia."
                ),
            ).tags.add(auto_tag)
            DocumentFactory(
                content=f"Factuur {i} voor verzekering, betaling van het formulier.",
            )

        train_and_save_classifier()
        classifier = DocumentClassifier()
        classifier.load()

        assert auto_tag.pk in classifier.predict_tags(
            "Verslag cardiologie: voorkamerfibrillatie, anticoagulantia.",
        )
        assert auto_tag.pk not in classifier.predict_tags(
            "Factuur voor de verzekering, betaling van het formulier.",
        )

    # Control: only national register numbers are removed. Lab values, dates
    # and phone numbers stay, also an international number with 11 digits,
    # so the masking costs little utility.
    def test_other_numbers_are_kept(self):
        text = "CRP 45 mg/L op 15/07/2026, bel 0470 12 34 56 of +32470123456."

        processed = DocumentClassifier().preprocess_content(text)

        kept = ["45", "15", "07", "2026", "0470", "12", "34", "56", "32470123456"]
        for number in kept:
            assert number in processed.split(), processed


class TestIssue6Erasure:
    def test_trashed_document_removed_from_model(
        self,
        classifier_model_file: Path,
        letter_gp_a: Document,
        letter_gp_b: Document,
        auto_tag: Tag,
        monkeypatch: pytest.MonkeyPatch,
        django_capture_on_commit_callbacks,
    ):
        letter_gp_b.tags.add(auto_tag)
        train_and_save_classifier()
        # Run the scheduled training at once instead of after the countdown
        cache.delete(CLASSIFIER_RETRAIN_PENDING_KEY)
        monkeypatch.setattr(
            tasks.train_classifier,
            "apply_async",
            lambda *args, **kwargs: tasks.train_classifier(),
        )

        with django_capture_on_commit_callbacks(execute=True):
            letter_gp_a.delete()  # soft delete: moves it to the trash

        # The saved model must be the one trained on the remaining documents.
        saved = DocumentClassifier()
        saved.load()
        expected = DocumentClassifier()
        expected.train()
        assert saved.last_auto_type_hash == expected.last_auto_type_hash
