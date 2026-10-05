# Frozen fine readout: complete original DEV241 from raw DINO

The unchanged frozen fine64 readout has a measurable increment over the same-lambda64 coarse readout on original DEV241, but its complete advantage over FoRIS, RCG, MEAN and fine16 is unresolved. This view preserves all original241 identities/order/keys; it does not select a favorable subset or refit the1200 producer.

COCO-20i exposed DEV241,79 classes,239 connected-photo groups, seed0, one reference,1024 working pixels. Class mIoU uses per-class summed I/U; paired95% intervals use2000 RandomState(0) connected support/query photo draws. All own packed snapshots were sealed before opening query GT for diagnostics. There were no new encoder forwards or inferred masks in this accounting run. The underlying fine method's recorded four shifted-query forwards remain part of its producer cost.

| Complete arm | mIoU |
|---|---:|
|Actual raw unprojected l24 NN origin|42.903888|
|Canonical complete FoRIS native|59.074825|
|RCG|61.019669|
|MEAN|60.680117|
|Coarse RCG64 control|60.128680|
|Fine RCG16 control|61.500661|
|Frozen primary fine RCG64|60.515912|
|Released INSID3 complete CRF control|55.006020|

| Frozen primary comparison | Gain [paired95% CI] | Interpretation |
|---|---:|---|
|vs actual raw NN|+17.612024 [14.236231,21.138078]|Clear complete-mask improvement over raw matching|
|vs complete FoRIS|+1.441087 [-0.754819,3.052193]|Unresolved; stable+2 is not established here|
|vs RCG|-0.503758 [-2.354283,0.733948]|Unresolved|
|vs MEAN|-0.164205 [-2.185183,1.038070]|Unresolved|
|vs same-lambda64 coarse control|+0.387231 [0.095634,0.753852]|Positive conditional fine-readout increment|
|vs fine16|-0.984750 [-2.976998,0.220770]|Unresolved; primary was not replaced after this result|
|vs released INSID3 CRF|+5.509891 [0.819143,7.933802]|Positive in this implementation/resource setting|

Fine64 versus native fold0/1/2/3 gains are+1.106391/+0.637296/+0.103814/+4.047149. Its same-lambda64 fine increment is positive in each fold (+0.674401/+0.295083/+0.064354/+0.521816). The large complete gain in fold3 primarily accompanies the coarse64 field; the fine increment is smaller. This diagnostic view does not alter the active frozen4000 recipe or its primary arm.

From the **actual raw NN origin**, primary fine64 adds4,127,069 true and2,503,298 false pixels, while deleting1,658,440 true and12,917,230 false pixels. Relative to **complete native separately**, it adds902,363 true and837,350 false pixels, while deleting994,107 true and2,862,840 false pixels. Thus native-relative intersection decreases91,744 pixels while union decreases2,025,490 pixels. These pooled counts describe the tradeoff; they are not a class-macro causal decomposition. Per-class/batch counts and all per-episode I/U/edits are retained.

Source/identity verification passed for every case:

- Frozen1200 matches use `(class,supportbasename,querybasename)`, not episode keys; all241 original rows and photo groups are retained.
- Canonical original packet/family native and RCG equal the mapped1200 masks bit for bit; old and new packet truth are pixel-identical; all six arm I/U pairs match the original1200 ledger.
- Every origin equals the sealed `stage:model.raw_nn` mask exactly. Parent protocol, stage config and original layer-producer seal bind it to normalized final-layer tokens before projection, serialized as float32. Other GT-fitted family-choice outputs are unused.
- Original GPU native replay differs by25 pixels in two cases (`2_8_26`:20, `1_37_1`:5). It is retained as `native.forward_replay.control`, scoring59.074810; its primary comparison is+1.441102 [-0.754826,3.052188]. Canonical native was not silently replaced.
- INSID3 uses the actual sealed `insid3.release_crf.control`, with matching family parent and GT-packet hash receipts. The cached stage INSID3 arm is excluded. Released logic uses the common timm DINOv3-L adapter/weights, paired BF16 extraction and640 CRF; hub numerical parity remains unverified.

Local independent checks reproduced every arm score and fold score, every raw/native four-edit I/U reconstruction and aggregate count, the original bootstrap draws, and all primary intervals using vectorized group/class I/U totals. Maximum interval discrepancy was2.09e-14. No efficacy-based subset, tuning, new component, source mutation, feature deletion or GPU invocation occurred.

Immutable snapshot `launch/frozen_fine_raw_dev241_v1`; producerPID46868/start954028662. Assembly3.536s, CPU scoring6.029s with four workers. Code SHA256 `efcf5154f7c4d67be3272bb32041ff21c23ce28a67000fdfc4dcf44077e81af0`; statistics module SHA256 `d95459d0b59c7048618661e7a16999248990b02e92cd1269d4629a497f62f843`.
