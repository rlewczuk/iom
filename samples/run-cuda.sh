#!/usr/bin/env bash
source /home/rlew/cuda_env.sh && exec ./build-cuda/iom_generate --model-dir /home/rlew/models/TinyLlama-1.1B-Chat-v1.0 --backend cuda --device 0 --tensor-arena-bytes 4294967296 --max-new-tokens 16 --message user "What is the capital of Poland ?"
