#!/usr/bin/env bash
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-sample-sycl-setvars.log 2>&1 || exit; set -u
cmake -S . -B build-sycl -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=ON && cmake --build build-sycl --target iom_generate -j
