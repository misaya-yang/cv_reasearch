# Matched official SAM3 comparison

Completed 2026-10-03 (America/New_York). COCO-20i / native val2014 JPEGs, seed0,
first50 author episodes per fold,200 total. Original instance annotation boxes
and the complete intervening3800 global box RNG draws were preserved. The metric
is the mean of the four fold-specific class-summed intersection/union macros.

| Arm | Class-mIoU | Paired difference from standard | 95% interval |
|---|---:|---:|---:|
| Original standard visual-only evaluator |59.9330|0|[0,0]|
| Previous pure frontend, identical instance boxes and author backend |59.9330|0|[0,0]|
| Original positive-mining evaluator, max_neg=0 |56.2190|−3.7140|[−5.4547,−0.4321]|

Standard and the matched previous frontend have **200/200 pixel-identical masks**,
with0 differing pixels. This validates those prompt/canvas/output frontends
under the same author processor. It does not independently compare the older
Meta implementation/MLP repair or the earlier semantic-component box sampler.

Fold0/1/2/3 standard:63.1010 /61.0109 /51.1180 /64.5022.
Positive mining:60.2710 /57.5631 /50.2338 /56.8081.
Bootstrap:2000 paired resamples of198 connected Source/Query photograph groups,
largest group2. These200 episodes do not establish the full4000 paper score or
close the SAM3 field; they measure the stated code comparison and this particular
mining procedure on the fixed subset.

One model build, three complete200-case arms, no training/download or category
text. Model load10.8129s; inference246.5321s; final report2.2914s;
guard requested provider shutdown at257.8972s. AutoDL subsequently showed E58
STOPPED. No-card restart was used to retrieve evidence and prepare the separately
requested QK method, without another paid GPU run.

Pinned author commit:b82ba838678ae8c2f27f618d9500a50c171e223e.
Script:`scripts/sam3_author_compare200.py`.
Evidence:`author200/report.json`, `execution.json`, `episodes.jsonl`, packed
masks, `guard.json`, and the immutable source/asset manifests.
