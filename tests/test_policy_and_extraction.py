import unittest

from mib_solution.contracts import blank_record, normalize_flags
from mib_solution.classical import Span, candidate_values, normalize_record, pick_fields
from mib_solution.extract import extract
from mib_solution.ocr import OCRPage
from mib_solution.policy import apply_safety_policy


class PolicyTests(unittest.TestCase):
    def test_disqualifying_risk_overrides_other_evidence(self):
        record = blank_record("MIB-123456")
        record.update({"visa_class": "DIP-1", "fee_status": "paid", "risk_flags": "active_warrant"})
        result = apply_safety_policy(record)
        self.assertEqual(result.decision, "DENIED")
        self.assertEqual(result.reason, "disqualifying_risk")

    def test_diplomatic_fee_waiver_is_not_an_override(self):
        record = blank_record("MIB-123456")
        record.update({"visa_class": "DIP-1", "fee_status": "waived"})
        self.assertIsNone(apply_safety_policy(record).decision)

    def test_unverified_waiver_requires_review(self):
        record = blank_record("MIB-123456")
        record.update({"visa_class": "XW-2", "fee_status": "waived"})
        self.assertEqual(apply_safety_policy(record).decision, "NEEDS_REVIEW")

    def test_missing_non_diplomatic_sponsor_requires_review(self):
        record = blank_record("MIB-123456")
        record.update({"visa_class": "XW-2", "fee_status": "paid"})
        self.assertEqual(apply_safety_policy(record).decision, "NEEDS_REVIEW")

    def test_diplomatic_record_is_not_denied_only_for_revoked_sponsor(self):
        record = blank_record("MIB-000002")
        record.update({"visa_class": "DIP-1", "sponsor_id": "SPN-0007", "fee_status": "paid"})
        self.assertIsNone(apply_safety_policy(record).decision)

    def test_unpaid_is_not_collapsed_into_paid(self):
        record = blank_record("MIB-000003")
        record["fee_status"] = "unpaid"
        normalize_record(record)
        self.assertEqual(record["fee_status"], "unpaid")

    def test_uppercase_unknown_visa_requires_review(self):
        record = blank_record("MIB-000101")
        record.update({"visa_class": "UNKNOWN", "fee_status": "paid", "sponsor_id": "SPN-1903"})
        self.assertEqual(apply_safety_policy(record).decision, "NEEDS_REVIEW")

    def test_transit_visa_still_denied_after_normalize(self):
        record = blank_record("MIB-000166")
        record.update({"visa_class": "TRANSIT-7", "fee_status": "paid", "sponsor_id": "SPN-7196"})
        normalize_record(record)
        self.assertEqual(apply_safety_policy(record).decision, "DENIED")

    def test_stale_non_diplomatic_arrival_is_denied(self):
        record = blank_record("MIB-000040")
        record.update({"visa_class": "XW-2", "sponsor_id": "SPN-1042", "fee_status": "paid", "arrival_date": "2025-12-01"})
        self.assertEqual(apply_safety_policy(record).decision, "DENIED")

    def test_review_only_risk_cannot_be_auto_approved(self):
        record = blank_record("MIB-123456")
        record.update({"visa_class": "DIP-1", "fee_status": "paid", "risk_flags": "rescinded_denial"})
        self.assertEqual(apply_safety_policy(record).decision, "NEEDS_REVIEW")


class ExtractionTests(unittest.TestCase):
    def test_receipt_amount_and_waiver_override_struck_obscured_status(self):
        spans = [
            Span("MIB Fee Receipt", 1, 0, 0, 1, 1, 10, 0, "text_layer"),
            Span("Fee Status", 1, 0, 10, 1, 11, 10, 0, "text_layer"),
            Span("[FEE STATUS OBSCURED]", 1, 20, 10, 40, 11, 10, 0, "text_layer"),
            Span("Amount", 1, 0, 20, 1, 21, 10, 0, "text_layer"),
            Span("$809.00", 1, 20, 20, 40, 21, 10, 0, "text_layer"),
            Span("Waiver Code", 1, 0, 30, 1, 31, 10, 0, "text_layer"),
            Span("N/A", 1, 20, 30, 40, 31, 10, 0, "text_layer"),
        ]
        candidates, _, _ = candidate_values(spans)
        record = blank_record("MIB-123456")
        pick_fields(record, candidates)
        normalize_record(record)
        self.assertEqual(record["fee_status"], "paid")

    def test_ocr_fee_debris_does_not_drop_paid_status(self):
        spans = [
            Span("MIB Fee Receipt", 1, 0, 0, 1, 1, 10, 0, "ocr:original"),
            Span("Fee Status: paid P", 1, 0, 10, 40, 11, 10, 0, "ocr:original"),
        ]
        candidates, _, _ = candidate_values(spans)
        record = blank_record("MIB-000093")
        pick_fields(record, candidates)
        normalize_record(record)
        self.assertEqual(record["fee_status"], "paid")

    def test_ocr_fee_typos_map_to_canonical_status(self):
        for raw, expected in (("Fee Status pag", "paid"), ("Fee Status waved", "waived"), ("Fee Status: paig", "paid")):
            spans = [
                Span("MIB Fee Receipt", 1, 0, 0, 1, 1, 10, 0, "ocr:original"),
                Span(raw, 1, 0, 10, 40, 11, 10, 0, "ocr:original"),
            ]
            candidates, _, _ = candidate_values(spans)
            record = blank_record("MIB-000012")
            pick_fields(record, candidates)
            normalize_record(record)
            self.assertEqual(record["fee_status"], expected, raw)

    def test_receipt_zero_amount_fallback_without_waiver_line(self):
        spans = [
            Span("MIB Fee Receipt", 1, 0, 0, 1, 1, 10, 0, "ocr:original"),
            Span("Amount $0.00", 1, 0, 20, 40, 21, 10, 0, "ocr:original"),
        ]
        candidates, _, _ = candidate_values(spans)
        record = blank_record("MIB-000015")
        pick_fields(record, candidates)
        normalize_record(record)
        self.assertEqual(record["fee_status"], "waived")

    def test_receipt_809_alone_does_not_infer_paid(self):
        spans = [
            Span("MIB Fee Receipt", 1, 0, 0, 1, 1, 10, 0, "ocr:original"),
            Span("Amount $809.00", 1, 0, 20, 40, 21, 10, 0, "ocr:original"),
        ]
        candidates, _, _ = candidate_values(spans)
        self.assertFalse(candidates["fee_status"])

    def test_rescinded_denial_prose_is_detected(self):
        spans = [
            Span("Manual Adjudicator Note", 1, 0, 0, 40, 1, 12, 0, "text_layer"),
            Span("Prior denial stamp rescinded. Route to human review.", 1, 0, 20, 80, 21, 10, 0, "text_layer"),
        ]
        _, flags, _ = candidate_values(spans)
        self.assertIn("rescinded_denial", flags)

    def test_ocr_bichazard_typo_maps_to_biohazard_red(self):
        spans = [
            Span("FORM B-13: Biometric Scan Slip", 4, 0, 0, 40, 1, 12, 0, "ocr:embedded_psm11"),
            Span("Observed flags: bichazard_red", 4, 0, 20, 80, 21, 10, 0, "ocr:embedded_psm11"),
        ]
        _, flags, _ = candidate_values(spans)
        self.assertIn("biohazard_red", flags)

    def test_text_layer_arrival_wins_over_conflicting_ocr(self):
        spans = [
            Span("Planetary Registry Extract", 1, 0, 0, 40, 1, 12, 0, "text_layer"),
            Span("Arrival Date", 1, 0, 20, 20, 21, 10, 0, "text_layer"),
            Span("2026-03-19", 1, 30, 20, 60, 21, 10, 0, "text_layer"),
            Span("Arrival Date 2026-99-99", 2, 0, 20, 60, 21, 10, 0, "ocr:original"),
        ]
        # Force same source_rank path via registry vs generic OCR page: inject equal ranks
        candidates, _, _ = candidate_values(spans)
        # Simulate equal-rank conflict directly.
        candidates["arrival_date"] = [
            (5.0, "2026-03-19", "text_layer"),
            (5.0, "2025-01-01", "ocr:original"),
        ]
        record = blank_record("MIB-000036")
        pick_fields(record, candidates)
        self.assertEqual(record["arrival_date"], "2026-03-19")

    def test_ocr_visa_typos_map_to_transit(self):
        spans = [
            Span("FORM I-8090: Extraterrestrial Work Authorization Intake", 1, 0, 0, 40, 1, 12, 0, "ocr:original"),
            Span("Visa Class: TRANSIT7", 1, 0, 20, 40, 21, 10, 0, "ocr:original"),
        ]
        candidates, _, _ = candidate_values(spans)
        record = blank_record("MIB-000101")
        pick_fields(record, candidates)
        normalize_record(record)
        self.assertEqual(record["visa_class"], "TRANSIT-7")

    def test_visible_manual_correction_overrides_a_missing_sponsor_cell(self):
        spans = [
            Span("FORM I-8090: Extraterrestrial Work Authorization Intake", 1, 0, 0, 1, 1, 10, 0, "text_layer"),
            Span("Manual correction: sponsor is SPN-4705.", 1, 0, 20, 60, 21, 10, 0, "text_layer"),
        ]
        candidates, _, _ = candidate_values(spans)
        record = blank_record("MIB-123456")
        pick_fields(record, candidates)
        normalize_record(record)
        self.assertEqual(record["sponsor_id"], "SPN-4705")

    def test_sponsor_attestation_applicant_prose_is_recoverable(self):
        spans = [
            Span("Sponsor Attestation Letter", 2, 0, 0, 40, 1, 12, 0, "text_layer"),
            Span(
                "Sponsor SPN-4705 attests that Zara Quill is expected on Earth for medical consult.",
                2, 0, 20, 80, 21, 10, 0, "text_layer",
            ),
        ]
        candidates, _, _ = candidate_values(spans)
        record = blank_record("MIB-123456")
        pick_fields(record, candidates)
        normalize_record(record)
        self.assertEqual(record["applicant_name"], "Zara Quill")
        self.assertEqual(record["sponsor_id"], "SPN-4705")
        self.assertEqual(record["declared_purpose"], "medical consult")

    def test_cut_out_attestation_applicant_placeholder_is_ignored(self):
        spans = [
            Span("Sponsor Attestation Letter", 2, 0, 0, 40, 1, 12, 0, "text_layer"),
            Span(
                "Sponsor SPN-4705 attests that [NAME CUT OUT] is expected on Earth for research.",
                2, 0, 20, 80, 21, 10, 0, "text_layer",
            ),
        ]
        candidates, _, _ = candidate_values(spans)
        self.assertFalse(candidates["applicant_name"])

    def test_original_ocr_beats_preprocessing_retry(self):
        pages = [
            OCRPage(1, "clean", "Visa Class: XW-1 Sponsor: SPN-0007", 0.99),
            OCRPage(1, "original", "Visa Class: XW-2 Sponsor: SPN-1042", 0.40),
        ]
        fields, _ = extract(pages)
        self.assertEqual(fields["visa_class"], "XW-2")
        self.assertEqual(fields["sponsor_id"], "SPN-1042")

    def test_risks_are_normalized_and_sorted(self):
        pages = [OCRPage(1, "original", "BIOHAZARD RED and active warrant", 0.9)]
        fields, _ = extract(pages)
        self.assertEqual(fields["risk_flags"], "active_warrant|biohazard_red")
        self.assertEqual(normalize_flags({"b", "a", "a"}), "a|b")

    def test_flattened_ocr_stops_a_field_at_the_next_label(self):
        pages = [OCRPage(1, "original", "Applicant Name: Zed Zarnax Species Code: ORION_GRAYS Fee Status: paid", 0.9)]
        fields, _ = extract(pages)
        self.assertEqual(fields["applicant_name"], "Zed Zarnax")
        self.assertEqual(fields["species_code"], "ORION_GRAYS")


if __name__ == "__main__":
    unittest.main()
