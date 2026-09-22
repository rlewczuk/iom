#!/usr/bin/env bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -fopenmp" -DBUILD_TESTING=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build-cpu --target iom_generate -j
