# hypo-G

accelerate launch --config_file recipes/accelerate_configs/zero2.yaml src/open_r1/grpo.py --config recipes/hypoGen/config_demo.yaml

nohup accelerate launch --config_file recipes/accelerate_configs/zero2.yaml src/open_r1/grpo.py --config recipes/hypoGen/config_demo.yaml > train.log 2>&1 &

1. hypothesis generation 移植
2. verl （pip）/ trainer
3. data preprocess （prompt）
4. reward function
  - inference
  - novelty
  - practicality
