#!/usr/bin/env bash
source /home/rlew/rocm_env.sh && cmake -S . -B build-rocm -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DSYCL_ENABLED=OFF && cmake --build build-rocm --target iom_generate -j
