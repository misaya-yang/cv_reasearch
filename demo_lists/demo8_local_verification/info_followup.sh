#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/demo8_local_verification
while ! test -f results/information_readout_v1/report.json; do
  if test -f results/information_readout_v1/error.json; then
    exit 1
  fi
  sleep 10
done
exec /root/miniconda3/bin/python -u info_analyze.py --root results
