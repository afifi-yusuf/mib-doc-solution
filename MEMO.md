# Technical Memo — Classical Visible-Evidence Intake

## Approach

I built an offline CPU pipeline that treats **visible evidence** as authoritative and uses local Tesseract only when a page has no meaningful PDF text layer.

**Extraction.** Native PyMuPDF spans are filtered for visibility (reject white/tiny hidden text). Candidates are gathered with document-type ranks: intake form, species registry, sponsor attestation, fee receipt. OCR fills scan-only pages and never outranks a unique trusted text value. Cross-page conflicts on applicant name or sponsor raise review flags; near-duplicate OCR names (`Ixokesh`/`Ikokesh`) and glued label debris are cleaned rather than treated as identity conflicts. Manual corrections supersede prior names.

**Adjudication.** Deterministic policy implements the field manual: disqualifying risk flags and unpaid fees deny; transit visas deny; revoked sponsors deny unless DIP-1; missing fee/visa evidence and unverified non-DIP waivers go to review. Approvals that depend on attestation for visa/sponsor require text-layer intake fields (species, home world, arrival), so OCR-completed supporting scans cannot approve over silent stamped risks.

**Calibration.** Fixed confidences by decision (`APPROVED` / `DENIED` / `NEEDS_REVIEW`) keep Brier loss stable without a learned calibrator.

No LLM, VLM, cloud OCR, or network is used at train or runtime.

## What worked

- Ranking + conflict rules recovered most clean digital packets.
- Fee OCR mid-band crops and typo repairs (`pal`→`paid`) fixed a batch of soft misses without inventing unpaid denials.
- Identity-conflict cleanup (name canonicalization, near-duplicates, manual-correction precedence) was the last clear train gain (~+1.8 points).
- Staying conservative on silent/illegible evidence preserved a CFA count of 18 on public train.

Public train: **~117.7 / 150** (extraction ~39.3, classification ~63.6, calibration ~14.9).

## Failure modes

- **Illegible or missing fee pages** dominate remaining `APPROVED→NEEDS_REVIEW` (~68 cases). Stamp color/shape did not correlate usefully with fee or risk.
- **Silent risk stamps** with no recoverable text: gold often expects `NEEDS_REVIEW` or `DENIED`; inventing stamp classes from CV alone was unreliable, so the system prefers review over speculative deny/approve.
- **Waived + non-DIP** approvals require trusting a waiver the receipt does not prove; a `DIP-WAIVER` shortcut increased catastrophic false approvals and was reverted.
- **OCR sponsor/visa substitutions** can still force wrong denies or review when the text layer is absent.

## With another week

1. Targeted receipt reconstruction (deskew + adaptive threshold only when fee is unknown and a receipt page is detected).
2. A small offline stamp classifier trained only on public labels, gated so it can demote to review but never create CFAs alone.
3. Better multi-page sponsor reconciliation when attestation and intake disagree on SPN digits.
4. Per-field confidence from OCR quality for calibration headroom.

## Compliance

Docker image: Tesseract + Poppler + PyMuPDF/Pillow. Entrypoint: `input_dir output_path`, `--network none`, scratch under `/tmp`, 4 workers for the 4-vCPU scoring host.
