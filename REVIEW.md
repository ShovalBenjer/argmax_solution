# REVIEW.md — judging policy for code review on this repository

Reader: an agent (or human) about to judge a pull request, not write one.
Build facts live in AGENTS.md; this file says only what to reject, what to
skip, and how to verify. Kilo Code reads this file from the PR base branch,
so a branch must never change the criteria it is judged by.

## Severity calibration

- **Blocker** — do not merge until fixed: security issue (secret in diff,
  injection, unsafe deserialization, unpinned action), data corruption or
  loss, broken build or test suite, dependency with a known CVE, a weakened
  test oracle (assertion removed, exception swallowed, `pytest.skip` without
  justification), or a change that silently degrades routing/classification
  quality with no measurement attached.
- **Major** — should be fixed: correctness risk under realistic inputs, new
  logic without tests, changed public API or prompt contract with no note in
  the PR body, magic constants that control cost/quality trade-offs.
- **Minor** — fix if cheap: naming, duplication, doc drift.
- **Nit** — optional; label it as such and never block on it.

Nitpicking generated or vendored files is a review failure, not diligence.

## Paths to skip (no style or content comment)

- `assets/`, lockfiles, and dependabot version churn (review the version
  delta, not the file).
- Generated artifacts and notebook outputs under `nb/`; append-only ledgers
  under `state/` if present.
- `.github/workflows/` diffs that only reformat YAML — verify the trigger,
  permissions, and pinned versions instead.
- Files the PR did not touch (no drive-by comments).

## Verification expected

State what you actually verified, not what you assume:

- Read the full diff (say so). For diffs over ~400 lines, review per file
  and name which files you read fully vs. spot-checked.
- Ran the relevant tests locally, or cite the CI run by job name — not just
  "green". A test that passes vacuously (e.g. catching an exception and
  returning, as `jev/test_router.py` once did) is a Blocker, not a pass.
- For `jev/` changes: name the routing decision you observed (route chosen,
  complexity score, threshold) or say you did not exercise it.
- Sub-agent budget: at most one review sub-agent per PR. Reviewers do not
  spawn reviewers.

## Summary style

Lead with the verdict (approve / request changes / comment), then Blockers,
then Majors. One line per finding with file and line. Do not restate the
diff.
