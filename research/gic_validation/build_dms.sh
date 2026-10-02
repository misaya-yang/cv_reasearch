#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gic_validation
mkdir -p sources runtime prefix
tar -xzf build-sources.tgz -C sources
find sources -name '._*' -type f -delete
cp -a sources/pybind11/. sources/DMS/cpp/pybind/pybind11/
cmake -S sources/eigen-3.4.0 -B runtime/eigen -DCMAKE_INSTALL_PREFIX="$PWD/prefix" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
cmake --install runtime/eigen
cmake -S sources/ceres-solver -B runtime/ceres -DCMAKE_INSTALL_PREFIX="$PWD/prefix" -DCMAKE_PREFIX_PATH="$PWD/prefix" -DCMAKE_BUILD_TYPE=Release -DMINIGLOG=ON -DGFLAGS=OFF -DSUITESPARSE=OFF -DLAPACK=OFF -DUSE_CUDA=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=ON
cmake --build runtime/ceres -j4
cmake --install runtime/ceres
cmake -S sources/DMS/cpp -B runtime/dms -DCMAKE_PREFIX_PATH="$PWD/prefix" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/root/miniconda3/bin/python -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python
cmake --build runtime/dms -j4
find runtime/dms -name 'pysumopt*.so' -exec cp '{}' env/ ';'
PYTHONPATH="$PWD/env:$PWD/sources/DMS/src" /root/miniconda3/bin/python -c 'import pysumopt; print("OFFICIAL_DMS_BUILD_IMPORT_PASSED",pysumopt.__file__)'
