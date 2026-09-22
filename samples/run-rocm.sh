#!/usr/bin/env bash
source /home/rlew/rocm_env.sh && exec ./build-rocm/iom_generate --model-dir /home/rlew/models/TinyLlama-1.1B-Chat-v1.0 --backend rocm --device 0 --tensor-arena-bytes 4294967296 --max-new-tokens 16 --message user "What is the capital of Poland ?"
