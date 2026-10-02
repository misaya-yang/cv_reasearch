# Repository layout and conventions

## Layout

```
AGENTS.md                      rules for agents
README.md                      one-screen map for people
docs/harness/                  operating documents for agents (indexed in INDEX.md)
docs/reference/                literature table, data audits
.claude/skills/                reusable skills (.agents/skills links here)
demo_lists/<direction>/        one directory per research direction
```

Nothing else at the root. No per-agent working folders, diaries, generated plans or exported copies of results.

## A direction directory

```
demo_lists/demoN_<slug>/
  README.md      numbers and conclusions, every number with its provenance
  HANDOFF.md     the method, how the idea arose, measured / inferred / unverified, tried and failed, pitfalls
  PLAN.md        next experiments: four lines, command, cost, stop rule
  <package>/     importable code
  scripts/       one entry point per experiment; the docstring says what it tests and what it writes
  results/       small result files, named <experiment>_f<fold>.json
```

- Currently: `demo9_transductive_ics` (live) and `demo4_incontext_seg` (its support library and error ledger;
  frozen, add no new experiments there).
- Server mirror: `/root/autodl-tmp/<directory name>`. Paths go through one `_paths.py` with environment
  overrides, not hard-coded per script.
- Direction documents may be in Chinese (the user reads them). One language per file.

## What is versioned

- Code, documents, and text results under `results/` (`.json`, `.log`, `.csv`, `.txt`), each under 1 MB.
- Never: weights, feature tensors, datasets, images, archives, third-party checkouts, environments. See
  `.gitignore`. Record a third-party dependency as URL plus revision in the direction's README.
- Commit only when the user asks.

## Life of a direction

1. **Open** only if all hold: the field has an oral or is clearly hot; a strong baseline with public code is
   reproduced locally; one test takes minutes on the shared GPU; an oracle experiment shows at least 5 points
   of headroom; you can name an information source or mechanism the baseline does not use. Spend at most one
   day on the quantified check, then ask the user.
2. **Live**: listed in `STATUS.md` with gates. Each gate has a date and a consequence.
3. **Close** when a stop rule is met: add the five-line note to `ARCHIVE.md` (idea, measured reason, what is
   kept, what not to retry, commit and path), delete the directory, clean the server, update `STATUS.md`.
   The code stays in git history.

## Editing shared files

Several agents work here at once. Re-read a shared file (`STATUS.md`, a direction's README) just before
editing it, change only your own section, and do not rewrite or reformat another agent's files.
