# Submit checklist

Local packaging is done. Finish these steps once GitHub auth works and validation predictions finish.

## 1. Publish the solution repo

```bash
cd ~/Desktop/mib-doc-solution
gh auth login -h github.com
gh repo create afifi-yusuf/mib-doc-solution --public --source=. --remote=origin --push
```

If the repo already exists:

```bash
git remote add origin git@github.com:afifi-yusuf/mib-doc-solution.git
git push -u origin main
```

## 2. Copy validation predictions into the challenge fork

After `/tmp/mib_validation_predictions.jsonl` reaches 5000 lines:

```bash
python3 ~/Desktop/mib-doc-challenge/scripts/validate_submission.py \
  --submission /tmp/mib_validation_predictions.jsonl \
  --manifest ~/Desktop/mib-doc-solution/data/validation_manifest.csv \
  --require-complete

cp /tmp/mib_validation_predictions.jsonl \
  ~/Desktop/mib-doc-challenge/submissions/afifi-yusuf/predictions.jsonl
```

## 3. Open the challenge PR

```bash
cd ~/Desktop/mib-doc-challenge
git checkout -b submission/afifi-yusuf
git add submissions/afifi-yusuf
git commit -m "Add afifi-yusuf MIB Doc Challenge submission."
git push -u origin HEAD
gh pr create --title "Submission: afifi-yusuf" --body "$(cat <<'EOF'
## Summary
- Classical visible-evidence offline pipeline
- Validation predictions + memo + solution repo link

## Checklist
- [x] predictions.jsonl
- [x] MEMO.md
- [x] SUBMISSION.md
- [ ] Submission form completed
EOF
)"
```

## 4. Form

Complete: https://docs.google.com/forms/d/1ZLkHmTsYd9I87JL1sUyps2rPTe6ohEI_lTZ8Jjts6bw/viewform

## Already verified locally

- Unit tests pass
- Docker image `mib-submission` builds (~115 MiB)
- Offline `docker run --network none` smoke on 3 PDFs validates
- Public train score ~117.7 / 150
