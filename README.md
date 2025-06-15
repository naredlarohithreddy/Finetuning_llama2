# Finetuning_llama2
This repository contains the setup, configuration, and training code for fine-tuning the Meta LLaMA 2 model using:  Parameter-Efficient Fine-Tuning (PEFT) with LoRA  bitsandbytes for 4-bit quantization

🦙 Fine-Tuning LLaMA 2 with LoRA and Transformers
This repository demonstrates the setup, configuration, and training code for fine-tuning Meta’s LLaMA 2 model using:

✅ Parameter-Efficient Fine-Tuning (PEFT) with LoRA
✅ 4-bit quantization using bitsandbytes for memory-efficient training
✅ Supervised Fine-Tuning (SFT) for instruction alignment

📌 Project Objective
Large Language Models (LLMs) like LLaMA 2 are excellent at generating text but often need alignment to follow human instructions effectively.
This project applies instruction tuning via Supervised Fine-Tuning (SFT) with LoRA, a practical alternative to more complex and resource-intensive methods like RLHF.

📚 Background
Why Fine-Tune?

Fine-tuning improves a model’s ability to:
  Follow instructions accurately
  Align with user expectations
  Specialize in domain-specific tasks
  
Fine-Tuning Techniques:

  Supervised Fine-Tuning (SFT): Trains on input-output pairs (instructions and ideal responses).
  RLHF: Involves human-in-the-loop feedback with a reward model and reinforcement learning (e.g., PPO).
  DPO: A newer, simpler alternative to RLHF that optimizes for preferences directly.

In this project, we use SFT, which is powerful when the model has seen similar data before and provides a solid baseline.


🔧 Implementation Overview
Prompt Template (LLaMA 2):

  <s> [INST] <<SYS>>
  System prompt
  <</SYS>>
  User prompt [/INST] Model response </s>

This format aligns with LLaMA 2's tokenizer and training format.

We use a public, instruction-style dataset:
  mlabonne/guanaco-llama2-1k

LoRA (Low-Rank Adaptation):
Instead of fine-tuning all weights, we use LoRA to apply low-rank updates to specific transformer layers, saving memory and training time.

LoRA Formula:

  Δ𝑊= (𝛼/𝑟)*𝐴/𝐵
  lora_alpha scales the update.
  Only the low-rank matrices A and B are trainable.
  Base model weights remain frozen.

🧪 Features

  ⚙️ Lightweight fine-tuning using LoRA adapters.
  🧠 Instruction formatting with LLaMA 2 prompt schema.
  📦 Integration with Hugging Face Transformers, Datasets, PEFT, and Accelerate.
  ✅ Easily extensible to support QLoRA or full parameter fine-tuning.


📁 Files
  fine_tuning_llama2.ipynb  – Main Jupyter notebook containing code, explanations, and experiments.

📈 Results
  We demonstrate how SFT using LoRA performs well for aligning LLaMA 2 with instruction-following tasks using a custom dataset. Output quality and model alignment improve significantly post fine-tuning.
