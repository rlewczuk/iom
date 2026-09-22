#!/usr/bin/env bash
exec ./build-cpu/iom_generate --model-dir /home/rlew/models/TinyLlama-1.1B-Chat-v1.0 --backend cpu --device 0 --max-new-tokens 1 --message user "What is the capital of Poland ?"
