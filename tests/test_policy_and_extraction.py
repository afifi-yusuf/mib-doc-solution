import unittest

from mib_solution.contracts import blank_record, normalize_flags
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

    def test_review_only_risk_cannot_be_auto_approved(self):
        record = blank_record("MIB-123456")
        record.update({"visa_class": "DIP-1", "fee_status": "paid", "risk_flags": "rescinded_denial"})
        self.assertEqual(apply_safety_policy(record).decision, "NEEDS_REVIEW")


class ExtractionTests(unittest.TestCase):
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
