#!/bin/bash
source /media/volume/bert_training_data_models/llama/reann/bin/activate
cd /media/volume/bert_training_data_models/llama/codes/extraction_framework
python3 benchmark_data/run_sdrf_benchmark.py --skip-conversion >> benchmark_data/benchmark_run.log 2>&1
echo "BENCHMARK FINISHED" >> benchmark_data/benchmark_run.log
