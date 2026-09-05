**Status:** done

## Summary

`SafeTensorsDir` now rejects duplicate tensor keys across shards with a `std::runtime_error` naming the new shard and the shard that supplied the kept entry. The task adds the diagnostic-only shard-path map and focused coverage for duplicate, three-way collision, disjoint ordering, and payload preservation.

## Verification

- `cmake -S . -B build/cc003 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cc003 -j --target iom_tests` — configured and built `iom_tests` successfully.
- `./build/cc003/test/iom_tests --test-case='SafeTensorsDir rejects duplicate keys across shards'` — 1 test case and 28 assertions passed.
- `g++ -std=c++20 -I/home/rlew/iom/src/iom/.work/0001-tensor-view/23-CC-001-validate-safetensors-payload-lengths/include /tmp/cc003_reproduce.cpp /home/rlew/iom/src/iom/.work/0001-tensor-view/23-CC-001-validate-safetensors-payload-lengths/build/cc001/libiom.a -pthread -o /tmp/cc003_reproduce && /tmp/cc003_reproduce` — uncorrected baseline reproduced `size=1 keys=2 payload=AAAA`.
- `cmake --build build/cc003 -j` — all configured CPU targets built successfully.
- `ctest --test-dir build/cc003 --output-on-failure -R '^iom_tests$'` — `iom_tests` passed.
- `ctest --test-dir build/cc003 --output-on-failure` — all 3 CPU test targets passed.
