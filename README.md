# Finetuning_llama2

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

&emsp;Follow instructions accurately

&emsp;Align with user expectations

&emsp;Specialize in domain-specific tasks
  
Fine-Tuning Techniques:

&emsp;Supervised Fine-Tuning (SFT): Trains on input-output pairs (instructions and ideal responses).

&emsp;RLHF: Involves human-in-the-loop feedback with a reward model and reinforcement learning (e.g., PPO).

&emsp;DPO: A newer, simpler alternative to RLHF that optimizes for preferences directly.

In this project, we use SFT, which is powerful when the model has seen similar data before and provides a solid baseline.


🔧 Implementation Overview

Prompt Template (LLaMA 2):

&emsp;`<s>` [INST] `<<SYS>>`  System prompt `<</SYS>>`

&emsp;User prompt [/INST] Model response `</s>`

This format aligns with LLaMA 2's tokenizer and training format.

We use a public, instruction-style dataset:

&emsp;mlabonne/guanaco-llama2-1k

LoRA (Low-Rank Adaptation):

**LoRA (Low-Rank Adaptation)** freezes the original weight matrix \( W_0 \) and learns a low-rank update:

$$
\Delta W = \frac{\alpha}{r} AB
$$

Instead of updating a full d×d weight matrix, LoRA decomposes it into two smaller matrices:

A∈R 
d×r
 
B∈R 
r×d
 

This reduces the number of trainable parameters from 
\( d^2 \)
  to 
\( 2dr \)
where 
𝑟
≪
𝑑
,making training significantly more efficient.
The number of additional parameters depends on the hyperparameter 
r, hence the name Low-Rank Adaptation.

&emsp;Instead of fine-tuning all weights, we use LoRA to apply low-rank updates to specific transformer layers, saving memory and training time.

### LoRA recap in one line

LoRA (Low-Rank Adaptation) freezes a large, pre-trained weight matrix **W₀** and learns a low-rank update

$$
\Delta W = \frac{\alpha}{r} \cdot A B
$$

that is **added** to the original weights during the forward pass.

---

## What **`lora_alpha`** does

| Symbol                 | Default role                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------- |
| **`r`**                | rank of the two small matrices $A\in\mathbb R^{d\times r},\;B\in\mathbb R^{r\times d}$ |
| **`α` (`lora_alpha`)** | *scaling factor* that controls the **magnitude** of the low-rank update                |

In code the forward pass of a LoRA-ised linear layer looks roughly like:

```python
def forward(x):
    original = x @ W0.T                     # frozen pre-trained weights
    lora_out = (x @ A) @ B                  # low-rank branch
    scaled   = (alpha / r) * lora_out       # <--  α/r  scaling
    return original + scaled                # add the adaptation
```

So :

1. The low-rank branch is computed.
2. It is multiplied by **`alpha / r`**.
3. The scaled result is **added to the main projection output** (for
   Q, K, V or FFN weights – wherever LoRA is inserted).

---

### Why use the scale?

* To keep the update’s variance comparable to the frozen weights.
* Original LoRA paper set **α = r** (so α/r = 1) by default; frameworks expose `lora_alpha` so you can dampen or amplify the adaptation:

  * **α < r** → gentler update, safer from overfitting.
  * **α > r** → stronger adaptation, may need smaller learning-rate.

---

lora_dropout is a regularisation knob inserted only on the LoRA branch to stop that low-rank update from over-fitting:

```python
drop    = nn.Dropout(p=lora_dropout)        # keeps p = 1-p_dropout tokens
lora_in = drop(x)                           # apply *after* layer-norm, before A
lora_out = (lora_in @ A) @ B
scaled   = (alpha / r) * lora_out
y        = original + scaled
```

What it does – randomly zero-outs rows of the mini-batch (or key/value heads, depending on the implementation) before they pass through the LoRA matrices
𝐴
,
𝐵
A,B.

Why – the backbone weights
𝑊
0
W
0
​
  stay intact; dropping activations only affects the learnable
Δ
𝑊
ΔW.
This nudges training to learn a low-rank update that is not overly reliant on any one token/head and keeps the adaptation small.

Typical default – lora_dropout = 0.0 (i.e., disabled) because LoRA already introduces relatively few new parameters; you enable it (e.g., 0.05 – 0.1) when you see the adaptation over-fitting on a small fine-tune set.

| Parameter      | Role in forward pass                                                             |
| -------------- | -------------------------------------------------------------------------------- |
| `lora_alpha`   | Scales the **magnitude** of the low-rank update $\alpha/r$                       |
| `lora_dropout` | Applies dropout **on the LoRA branch’s input** before $A$, adding regularisation |


### Where in the Transformer?

`A,B` blocks are typically attached to the **linear projections** inside self-attention (`W_Q`, `W_K`, `W_V`, `W_O`) and/or the feed-forward layers.
The scaling happens *inside those projections* – not at the entire attention output.

---

> **Bottom line:** `lora_alpha` is just a scalar that rescales the learned low-rank update before it is *added* to the frozen weight’s output. Bigger `alpha` ⇒ bigger influence of LoRA on the final activations.



🧪 Features

&emsp;⚙️ Lightweight fine-tuning using LoRA adapters.

&emsp;🧠 Instruction formatting with LLaMA 2 prompt schema.

&emsp;📦 Integration with Hugging Face Transformers, Datasets, PEFT, and Accelerate.

&emsp;✅ Easily extensible to support QLoRA or full parameter fine-tuning.


📁 Files

&emsp;fine_tuning_llama2.ipynb  – Main Jupyter notebook containing code, explanations, and experiments.

📈 Results

&emsp;We demonstrate how SFT using LoRA performs well for aligning LLaMA 2 with instruction-following tasks using a custom dataset. Output quality and model alignment improve significantly post fine-tuning.
