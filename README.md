# Improving creativity of LLM via pure Reinforcement Learning

Large Language Models (LLMs) exhibit strong performance across a variety of tasks, yet their outputs often remain confined to frequent patterns learned during training, limiting their utility for exploratory and innovative tasks such as scientific discovery and creative ideation. We address this challenge by proposing a novel reinforcement learning framework that enhances LLMs' ability to generate novel yet reliable hypotheses. Specifically, we integrate Random Network Distillation (RND) into the semantic output space of LLMs to quantify and reward semantic novelty, and further incorporate a semantic similarity-based metric to promote diversity. To ensure the factual reliability and coherence of generated hypotheses, we augment the training with Reinforcement Learning with AI Feedback (RLAIF) guided by soundness prompts. Our method is embedded within the Generalized Reinforcement Policy Optimization (GRPO) pipeline, ensuring stable training at scale. Experiments demonstrate that our approach significantly increases the novelty of generated hypotheses without sacrificing soundness, offering a promising direction for enhancing LLM-driven scientific and creative exploration.

## Before start
Follow installation step of Open-R1
Replace open-r1/src/grpo_trainer.py in trl package to enable RND training.


## Test run

accelerate launch --config_file recipes/accelerate_configs/zero2.yaml src/open_r1/grpo.py --config recipes/hypoGen/config_demo.yaml

nohup accelerate launch --config_file recipes/accelerate_configs/zero2.yaml src/open_r1/grpo.py --config recipes/hypoGen/config_demo.yaml > train.log 2>&1 &

## Our work

 -  We introduce the first known integration of RND into the semantic output space of LLMs, enabling models to explicitly measure and reward hypothesis novelty based on intrinsic prediction errors.

 -  We develop a composite reward strategy combining intrinsic RND-based novelty and semantic similarity-based novelty, effectively encouraging semantic exploration and generation diversity.

 -  We adopt and enhance the RLAIF framework by employing inference models and ``soundness prompts'' to systematically evaluate and maintain the factual reliability and logical coherence of generated hypotheses.

 -  We successfully implement and demonstrate our approach within a GRPO-based training framework, providing empirical evidence for improved novelty without compromising soundness.

## Citation
```
@misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {Hugging Face},
    month = {January},
    year = {2025}
}
```
