# WildVideo: Benchmarking LMMs for Understanding Video-Language Interaction

## Overview

<p align="center">
  <img src="images/Task_definition.png" alt="WildVideo overview" width="60%">
</p>

ildVideo is an open-world benchmark dataset designed to address how to assess hallucination of Large Multi-modal Models (LMMs) for understanding video-language interaction in the wild. 

### 1)  Multi-level, multi-aspect task design
We define 9 distinct tasks that stress-test LMMs across:
- **Perceptual tasks** – e.g., static perception, dynamic perception.  
- **Cognitive tasks** – e.g., commonsense reasoning, world knowledge reasoning.  
- **Contextual comprehension tasks** – e.g., contextual ellipsis, cross-turn retrieval.

### 2)  Open-world videos from two human perspectives
WildVideo is built from 1,318 carefully curated videos collected from open-world scenarios, covering:
- **First-person (ego-centric)**
- **Third-person (observer)**
On top of this video collection, we construct:
- **13,704 single-turn QA pairs**
- **1,585 multi-turn dialogues** - up to 5 turns each.
## Evaluation pipeline
The evaluation of WildVideo is integrated into [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main). The detailed instructions of the evaluation are shown as follows.
### Installation

For formal usage, you can install the package from PyPI by running the following command:
```bash
pip install lmms-eval
```

For development, you can install the package by cloning the repository and running the following command:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .
```

If you want to test LLaVA, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e .
```
### Evaluation

We use [LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) as an example in the following commands. You can change `--model`, and `--model_args` based on your requirement.

**Evaluation of LLaVA-Video-7B-Qwen2 on Wildvideo (single_en / single_cn)**

```bash
accelerate launch --num_processes=4 \
    -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
    --tasks wildvideo_single_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid \
    --output_path ./logs/
```
**Evaluation of LLaVA-Video-7B-Qwen2 on Wildvideo (multi_en / multi_cn)**

```bash
accelerate launch --num_processes=4 \
    -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
    --tasks wildvideo_multi_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid \
    --output_path ./logs/
```
