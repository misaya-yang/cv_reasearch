# Templates

Copy, fill in, keep short.

## Four-line card (before every run)

```
Tests:      <assumption, e.g. H2: reliability scores carry information about pseudo-reference quality>
Predicts:   <numbers, derived from measured quantities: top half - random half >= 2 points; bottom < random>
If matched: <what it means and the next run>
If not:     <which assumption is refuted and how the method changes>
Cost:       <samples, minutes, command>
Stop:       <when to abort early, e.g. no difference after 200 episodes>
```

## Error ledger

| Intervention (one decision replaced by ground truth) | Metric | Gain | Removable? |
|---|---:|---:|---|
| Baseline, reproduced locally (paper: ...) | | | |
| Oracle choice among the method's own candidates | | | |
| Only add what was missed | | | |
| Only remove false positives | | | |
| Learned scorer on available evidence, held-out classes | | | measuring device, not a method |

State the dataset, split, sample count and the script under the table.

## Result table

| Row | What it uses | Fold 0 | ... | Mean | Paired difference to baseline [95% interval] |
|---|---|---:|---:|---:|---|
| Baseline | | | | | |
| Naive, same information | | | | | |
| Method | | | | | |
| Random selection / inverted selection | | | | | |
| Oracle upper bound (uses ground truth) | | | | | |

Under the table: units resampled, number of bootstrap draws, seeds, how many units went up and down, script and
result file.

## Plan entry (one per experiment, for another agent)

```
## E<n> <name>
- Tests / Predicts / If matched / If not      (the four lines)
- Command: <exact command>, <time>, <disk>
- Start small: <first fold, first 200 units>
- Stop rule: <condition that ends the experiment early or closes the direction>
- Already tried, do not repeat: <list with numbers>
```

## Hand-off document

1. One sentence: what the method is and the main number with its interval.
2. The method as it stands, step by step.
3. How the idea arose: the measurements that led to it, in order.
4. Measured / inferred / unverified, three separate lists.
5. Tried and failed, with numbers.
6. Pitfalls that cost time.
7. Files: script, what it produces, which table it feeds.

## Closing note for a stopped direction (five lines)

```
Idea:      <one sentence>
Closed:    <date>, <measured reason with numbers>
Kept:      <what remains true and reusable: a diagnosis, a tool, a data audit>
Do not:    <what must not be retried, and what new evidence would justify reopening>
Location:  <commit hash and path of the code>
```

## Progress note to the user

Conclusion first. Then the numbers with their controls. Then what is unverified. Then what is needed from the
user, as a separate list. No result means say "no result"; a failure means say so with the numbers.
