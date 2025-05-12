# ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3 \
#     src/open_r1/grpo.py --config recipes/hypoGen/exp/zs_3b.yaml 

# ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3 \
#     src/open_r1/grpo.py --config recipes/hypoGen/exp/fs_3b.yaml 

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 1 \
    src/open_r1/grpo.py --config recipes/hypoGen/exp/zs_rnd_1s5b.yaml 