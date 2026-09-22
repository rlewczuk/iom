#!/usr/bin/env bash
source /home/rlew/cuda_env.sh && cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build-cuda --target iom_generate -j
