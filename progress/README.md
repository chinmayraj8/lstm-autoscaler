# Progress Log — how this folder works

This folder is the running record of every real change made to this project:
what we changed, why, and — most importantly — **how the actual numbers moved**.
It exists so nobody (including future-you, six weeks from now) has to
reconstruct "wait, why did we change this?" from memory or git blame.

## Rules

1. **One file per unit of work**, named `YYYY-MM-DD_short-slug.md`.
2. **Every entry follows the template below.** Don't skip sections — "Impact"
   and "Still open" are the two people actually come back to read.
3. **Never overwrite an old entry.** If a later step changes something an
   earlier entry claimed, add a new entry that says so and links back
   (`progress/00_INDEX.md` is not append-only, but the dated files are).
4. **`00_INDEX.md` gets one new row per entry**, added at the same time as
   the entry itself — newest at the top.
5. If a change doesn't move any number (pure refactor, cleanup, docs), say so
   explicitly rather than leaving "Before -> After" blank.

## Entry template

```markdown
# Step N: <short title>
Date: YYYY-MM-DD
Status: done | in-progress | blocked

## What changed
Plain description of the actual change (code, config, data).

## Why
What problem this addresses — link back to the audit section if relevant.

## Before -> After
| Metric | Before | After |
|---|---:|---:|
| ... | ... | ... |

## Impact
What this means in plain terms. Did a claim get stronger, weaker, or
disproven? Say so even if the answer is unflattering.

## Still open
What this step did NOT fix, so the next entry (or the next person) doesn't
assume more was done than actually was.
```

## Index

See `00_INDEX.md` for the full list, newest first.
