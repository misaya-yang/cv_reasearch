# Harness docs

Operating documents for agents. `AGENTS.md` at the repository root holds the rules; these files hold the detail.
Keep each file short and current. Add a row here when adding a file.

| File | Read it | Contents |
|---|---|---|
| [STATUS.md](STATUS.md) | at the start of every session | the live direction, what is measured, gates, who runs what, what the user must decide |
| [SERVER.md](SERVER.md) | before the first `ssh` | access, limits, shared assets, pitfalls, cleanup |
| [REPO.md](REPO.md) | before creating, moving or deleting files | layout, naming, what is versioned, how a direction opens and closes |
| [ARCHIVE.md](ARCHIVE.md) | before proposing an idea | closed directions: the measured reason, what is reusable, where the code is |

Related:

- Research method (skill): `.claude/skills/measure-first-research/SKILL.md`, with worked examples, templates
  and a paired-bootstrap script. `.agents/skills/` links to the same folder.
- Reference material: `docs/reference/` (literature table of 22 papers; COCO-20i image-overlap audit).
- Live direction: `demo_lists/demo9_transductive_ics/` (`HANDOFF.md`, `PLAN.md`, `README.md`).
