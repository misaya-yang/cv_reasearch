#!/usr/bin/env bash
# Prepare INSID3's in-context segmentation benchmarks (INSID3/docs/data.md) under $DATA on the server.
#
#   get_data.sh <step> [<step> ...]        steps: pascal_images suim lung isic coco2017_val
#                                                 coco_official lvis paco pascal_part isaid   (these five need Drive files)
#                                                 coco2017_train_subset                       (after lvis and paco)
#
# The server cannot reach Google Drive. Download the Drive archives elsewhere and copy them into $DATA/_drive
# under these names: coco_train2014_masks.zip coco_val2014_masks.zip coco_splits.zip lvis.zip paco.zip pascal.zip
# iSAID_5i.zip isaid_splits.zip (file ids are in INSID3/docs/data.md).
# A step is skipped when $DATA/.done_<step> exists. Archives are deleted after extraction.
# PerMIS is left out: it needs the 119 GB TAO test archive, more than the data disk holds.
set -uo pipefail
DATA=${DATA:-/root/autodl-tmp/datasets/ics}; DRIVE=$DATA/_drive; PUB=/root/autodl-pub
PY="env PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python"
mkdir -p "$DATA" "$DRIVE"
fetch() {                    # resumable; <file>.ok marks a finished download. Slow link: run under `source /etc/network_turbo`
  [ -e "$2.ok" ] && return 0
  curl -fL -C - --retry 8 --retry-delay 15 -sS -A "curl/8.0" -o "$2" "$1" && touch "$2.ok"
}
pfetch() {                   # pfetch <url> <file> <parts>: parallel ranges for hosts that throttle each connection (S3: 140 KB/s)
  [ -e "$2.ok" ] && return 0
  local size n=$3 i
  size=$(curl -sIL -A "curl/8.0" "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
  local chunk=$(( (size + n - 1) / n ))
  for i in $(seq 0 $((n - 1))); do
    (
      a=$((i * chunk)); b=$(( (i + 1) * chunk - 1 )); [ "$b" -ge "$size" ] && b=$((size - 1))
      part="$2.part$i"; want=$((b - a + 1)); touch "$part"; have=$(stat -c %s "$part")
      until [ "$have" -ge "$want" ]; do
        curl -fsSL -A "curl/8.0" -r "$((a + have))-$b" "$1" >> "$part" || sleep 5
        have=$(stat -c %s "$part")
      done
    ) &
  done
  wait
  : > "$2"
  for i in $(seq 0 $((n - 1))); do cat "$2.part$i" >> "$2"; rm -f "$2.part$i"; done
  [ "$(stat -c %s "$2")" = "$size" ] && touch "$2.ok"
}
need() { [ -s "$DRIVE/$1" ] || { echo "[missing] $DRIVE/$1"; return 1; }; }

pascal_images() {            # VOC 2010 trainval, 1.35 GB
  mkdir -p "$DATA/Pascal-Part"; cd "$DATA/Pascal-Part"
  fetch http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar voc2010.tar
  tar -xf voc2010.tar VOCdevkit/VOC2010/JPEGImages
  rm -f voc2010.tar voc2010.tar.ok
  echo "JPEGImages: $(ls VOCdevkit/VOC2010/JPEGImages | wc -l) (expected 11321)"
}
suim() {                     # 0.18 GB
  mkdir -p "$DATA/SUIM"; cd "$DATA/SUIM"
  fetch https://www.kaggle.com/api/v1/datasets/download/heyoujue/suim-merged suim-merged.zip
  unzip -q -o suim-merged.zip
  cp -r suim_merged/* .
  rm -rf suim_merged suim-merged.zip suim-merged.zip.ok
  echo "images: $(ls images | wc -l) (expected 1635)"
}
lung() {                     # 10.3 GB archive; only CXR_png and masks are kept
  mkdir -p "$DATA/LungSegmentation"; cd "$DATA/LungSegmentation"
  fetch https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels cxr.zip
  unzip -q -o cxr.zip 'Lung Segmentation/CXR_png/*' 'Lung Segmentation/masks/*'
  rm -f cxr.zip cxr.zip.ok
  mv 'Lung Segmentation'/* .
  rmdir 'Lung Segmentation'
  echo "CXR_png: $(ls CXR_png | wc -l) (expected 800), masks: $(ls masks | wc -l) (expected 704)"
}
isic() {                     # 11.2 GB archive; as in DR-Adapter's ISIC_Split.py: resize to 512x512 into class folders 1/ 2/ 3/
  mkdir -p "$DATA/ISIC"; cd "$DATA/ISIC"
  fetch https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip gt.zip
  pfetch https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip in.zip 32
  fetch https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/isic/class_id.csv class_id.csv
  unzip -q -o gt.zip
  unzip -q -o in.zip
  rm -f gt.zip in.zip gt.zip.ok in.zip.ok
  $PY - <<'EOF'
import csv, os
from PIL import Image
src = "ISIC2018_Task1-2_Training_Input"; fold = {"seborrheic_keratosis": "1", "nevus": "2", "melanoma": "3"}; n = {}
for d in "123": os.makedirs(os.path.join(src, d), exist_ok=True)
for row in csv.DictReader(open("class_id.csv")):
    f = os.path.join(src, row["ID"] + ".jpg")
    if not os.path.exists(f): continue
    Image.open(f).resize((512, 512)).save(os.path.join(src, fold[row["Class"]], row["ID"] + ".jpg"))
    os.remove(f); n[fold[row["Class"]]] = n.get(fold[row["Class"]], 0) + 1
print("ISIC folds:", dict(sorted(n.items())), "(expected 208 / 1867 / 519)")
EOF
  rm -f ISIC2018_Task1-2_Training_Input/*.jpg ISIC2018_Task1-2_Training_Input/*.txt class_id.csv class_id.csv.ok
  echo "masks: $(ls ISIC2018_Task1_Training_GroundTruth | grep -c segmentation) (expected 2594)"
}
coco2017_val() {             # images for LVIS-92i and PACO-Part; training images are added once the annotation files say which
  mkdir -p "$DATA/LVIS/coco"; cd "$DATA/LVIS/coco"
  unzip -q -n "$PUB/COCO2017/val2017.zip"
  echo "val2017: $(ls val2017 | wc -l)"
}
coco2017_train_subset() {    # LVIS-92i and PACO-Part validation episodes also use train2017 images; extract only those
  cd "$DATA/LVIS/coco"
  $PY - "$DATA" "$PUB/COCO2017/train2017.zip" <<'EOF'
import os, pickle, sys, zipfile
data, zpath = sys.argv[1], sys.argv[2]; need = set()
def walk(o):
    if isinstance(o, str):
        if "train2017/" in o and o.endswith(".jpg"): need.add("train2017/" + o.split("train2017/")[1])
    elif isinstance(o, dict):
        for k, v in o.items(): walk(k); walk(v)
    elif isinstance(o, (list, tuple, set)):
        for v in o: walk(v)
for f in ["LVIS/lvis_val.pkl", "PACO-Part/paco/paco_part_val.pkl"]:
    p = os.path.join(data, f)
    if os.path.exists(p): n0 = len(need); walk(pickle.load(open(p, "rb"))); print(f, "adds", len(need) - n0)
    else: raise SystemExit(f"missing {p}")
todo = sorted(n for n in need if not os.path.exists(n)); print("train2017 images needed", len(need), "to extract", len(todo))
with zipfile.ZipFile(zpath) as z:
    for n in todo: z.extract(n, ".")
print("train2017 now holds", len(os.listdir("train2017")))
EOF
}
coco_official() {            # official COCO-20i masks and splits, kept apart from the rebuilt masks in /root/demo4_cache
  need coco_val2014_masks.zip; need coco_train2014_masks.zip; need coco_splits.zip
  mkdir -p "$DATA/COCO2014/annotations"; cd "$DATA/COCO2014/annotations"
  unzip -q -o "$DRIVE/coco_val2014_masks.zip"
  unzip -q -o "$DRIVE/coco_train2014_masks.zip"
  cd ..
  unzip -q -o "$DRIVE/coco_splits.zip"
  rm -f "$DRIVE"/coco_*_masks.zip "$DRIVE/coco_splits.zip"
  echo "val masks: $(ls annotations/val2014 | wc -l), train masks: $(ls annotations/train2014 | wc -l)"
}
lvis() {
  need lvis.zip
  mkdir -p "$DATA/LVIS"; cd "$DATA/LVIS"
  unzip -q -o "$DRIVE/lvis.zip"
  mv lvis/* .
  rm -rf lvis "$DRIVE/lvis.zip"
  ls -la ./*.pkl
}
paco() {
  need paco.zip
  mkdir -p "$DATA/PACO-Part"; cd "$DATA/PACO-Part"
  ln -sfn ../LVIS/coco coco
  unzip -q -o "$DRIVE/paco.zip"
  rm -f "$DRIVE/paco.zip"
  ls -la paco
}
pascal_part() {
  need pascal.zip
  mkdir -p "$DATA/Pascal-Part/VOCdevkit/VOC2010"; cd "$DATA/Pascal-Part"
  unzip -q -o "$DRIVE/pascal.zip"
  cp -r pascal/* VOCdevkit/VOC2010/
  rm -rf pascal "$DRIVE/pascal.zip"
  ls VOCdevkit/VOC2010
}
isaid() {
  need iSAID_5i.zip; need isaid_splits.zip
  mkdir -p "$DATA/iSAID"; cd "$DATA/iSAID"
  unzip -q -o "$DRIVE/iSAID_5i.zip"
  cp -r iSAID_patches/* .
  rm -rf iSAID_patches "$DRIVE/iSAID_5i.zip"
  unzip -q -o "$DRIVE/isaid_splits.zip"
  rm -f "$DRIVE/isaid_splits.zip"
  echo "images: $(ls val/images | wc -l) (expected 6363)"
}

for step in "$@"; do
  if [ -e "$DATA/.done_$step" ]; then echo "[skip] $step"; continue; fi
  echo "[start] $step $(date +%H:%M)"
  ( set -e; "$step" ); rc=$?
  if [ $rc -eq 0 ]; then touch "$DATA/.done_$step"; echo "[done] $step $(date +%H:%M)"; else echo "[FAILED] $step (exit $rc)"; fi
done
df -h /root/autodl-tmp | tail -1
echo "[finished] $*"
