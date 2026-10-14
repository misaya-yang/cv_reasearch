# T8 prefix diagnosis (S half only)

The frozen tail/control sample contains 322 S-half cases; T labels were not read and no encoder calls were made.
Rates are fractions of selected error-region pixels classified foreground at each cached output prefix. They are a targeted diagnostic, not dataset-wide accuracy.

In the tail group, FoRIS `pre` foreground rates are 0.046–0.070 in candidate-only false-positive pixels and 0.912–0.964 in candidate-missed true-positive pixels; the final FoRIS mask is 0 and 1, respectively. The candidate `c_mod+V+P` prefix marks 0.077–0.381 of the latter pixels foreground, while the final candidate mask marks none. This shows disagreement already present in the frozen prefix masks for these selected cases; it does not establish a general correction rule.

|Domain|Group|Region|Images|Pixels|FoRIS pre|FoRIS final|Own c_mod+V+P|Own no-e|Candidate final|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|coco|tail|E_FP|46|3116184|0.050|0.000|0.828|0.946|1.000|
|coco|tail|E_FN|46|1612128|0.912|1.000|0.180|0.072|0.000|
|coco|control|E_FP|46|171566|0.325|0.000|0.705|0.907|1.000|
|coco|control|E_FN|46|271010|0.790|1.000|0.093|0.098|0.000|
|isic|tail|E_FP|31|2054062|0.056|0.000|0.611|0.934|1.000|
|isic|tail|E_FN|31|2582760|0.953|1.000|0.077|0.069|0.000|
|isic|control|E_FP|31|160876|0.238|0.000|0.633|0.628|1.000|
|isic|control|E_FN|31|262041|0.875|1.000|0.191|0.101|0.000|
|suim|tail|E_FP|65|6101763|0.046|0.000|0.678|0.858|1.000|
|suim|tail|E_FN|65|6843669|0.946|1.000|0.207|0.186|0.000|
|suim|control|E_FP|65|310621|0.292|0.000|0.480|0.841|1.000|
|suim|control|E_FN|65|895749|0.919|1.000|0.371|0.281|0.000|
|lvis|tail|E_FP|19|804344|0.070|0.000|0.582|0.690|1.000|
|lvis|tail|E_FN|19|3207300|0.964|1.000|0.381|0.392|0.000|
|lvis|control|E_FP|19|147436|0.355|0.000|0.444|0.630|1.000|
|lvis|control|E_FN|19|205998|0.761|1.000|0.226|0.066|0.000|

Selection SHA256: `8a2fbc9e0762d4ce56e3ae37e8b56f0e00da6aedebcad8967d0617826cb5d0c3`.
Per-case records: `/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1013_T8/descriptions.jsonl`; case sheets: `/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1013_T8/cases`.
