# Technical Memo — Classical Visible-Evidence Intake

## Problem framing

MIB packets mix clean digital forms, scan-only pages, sponsor prose, fee receipts, and inspection stamps. The text layer is not trustworthy: white/tiny spans can inject false fields or “instructions.” Scoring also punishes catastrophic false approvals much harder than cautious `NEEDS_REVIEW`, so the design goal was **evidence-bound extraction + conservative adjudication**, not maximum field recall.

## Approach

**Visible-first extraction.** PyMuPDF spans are kept only when they look like real ink (reject white/near-white and tiny type). Candidates are scored by document role — intake → registry → attestation → fee — so a supporting letter cannot silently overwrite the intake form. OCR runs only on pages without meaningful visible text; a unique trusted text value always wins over OCR at the same rank.

**Conflict as a feature.** Disagreeing applicant names or sponsor IDs become review flags rather than arbitrary picks. That surfaced a real engineering bug: OCR debris (`Ixokesh` vs `Ikokesh`, trailing `|`, glued `PASSPORT IMAGE` / next-row labels) was inventing false `identity_conflict`. Fixes that paid off were boring but specific — name canonicalization, edit-distance near-duplicates, stripping gutter punctuation, and treating `Manual correction: applicant is …` as authoritative over pre-correction text.

**Attestation is useful and dangerous.** Sponsor letters often carry the only clean prose for name, purpose, or visa when the intake scan is wrecked. Early on, filling visa/sponsor from attestation while OCR “completed” species/home world looked like a full packet and produced an approve — including over a silent `active_warrant` page with no recoverable text. The gate now requires **text-layer** intake fields before an attestation-backed approve. That is the kind of bug you only catch by tracing CFA diffs, not by reading the field manual.

**Policy is explicit and boring on purpose.** Hard denies for disqualifying flags, unpaid fees, transit visas, and revoked sponsors (with the DIP-1 carve-out). Missing fee/visa → review. Non-DIP `waived` → review unless the packet independently proves diplomacy. Fixed confidences by decision class keep calibration stable without a second model.

Runtime is local Tesseract + Poppler + PyMuPDF/Pillow only. No LLM/VLM, no cloud OCR, no network.

## Things we noticed (negative results matter)

- **Silent stamps are a trap.** Several gold denies/reviews have no readable risk wording. Organizer clarification matched what the data shows: when the flag is visual-only, `NEEDS_REVIEW` is the honest answer. Color/shape heuristics and stamp CNNs did not produce a safe signal; blue/red/wax stamps also did not correlate with fee status or adjudication in a full-train audit.
- **Fee pages are often unreadable, not under-parsed.** Mid-band crops and typo repairs (`pal`→`paid`, bare `Status:` lines) recovered a real batch, then returns collapsed. Remaining `paid/waived → unknown` soft misses are mostly missing or illegible receipts — spending more OCR budget there mostly created noise.
- **“Clever” waiver shortcuts hurt.** Inferring approve from a visible `DIP-WAIVER`-style code looked promising and raised catastrophic false approvals. Reverted. Unverified non-DIP waivers stay in review.
- **False identity conflict was the last clean win.** Cleaning OCR name conflicts recovered real `APPROVED` cases without opening a CFA hole, once the attestation intake gate was tightened.
- **Classification > extraction in the score.** A pipeline that invents fees or stamps to chase extraction points will lose on CFAs. Prefer unknown + review.

## Failure modes still open

Illegible fee receipts; OCR digit substitutions on sponsor IDs (e.g. reading a revoked `SPN-4040` instead of the true sponsor); waived non-DIP packets that gold somehow approves; transit/unpaid cases where the decisive token never OCRs cleanly.

## With another week

Receipt-only enhancement when fee is unknown; a **review-only** stamp head that may demote approve→review but never deny/approve alone; sponsor digit reconciliation across pages; OCR-quality-aware confidence for the calibration slice.

## Compliance note

Docker entrypoint: `<input_pdf_dir> <output_predictions_path>`, `--network none`, scratch under `/tmp`, four workers for the four-vCPU host.
