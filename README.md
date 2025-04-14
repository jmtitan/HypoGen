# hypo-G

`CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct`

`accelerate launch --config_file recipes/accelerate_configs/2gpu_config.yaml --num_processes 1 src/open_r1/grpo.py --config recipes/hypoGen/config_demo.yaml`

1. hypothesis generation 移植
2. verl （pip）/ trainer
3. data preprocess （prompt）
4. reward function
  - inference
  - novelty
  - practicality
