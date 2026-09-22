#!/usr/bin/env bash
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-sample-sycl-setvars.log 2>&1 || exit; set -u
exec ./build-sycl/iom_generate --model-dir /home/rlew/models/TinyLlama-1.1B-Chat-v1.0 --backend sycl --device 0 --tensor-arena-bytes 4294967296 --max-new-tokens 16 --message user "What is the capital of Poland ?"
