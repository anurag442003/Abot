# The Price Is Right 💰🤖

The Price Is Right is an end-to-end machine learning system that predicts the retail price of a product solely from its text description. Given a product's title, category, brand, and feature details, the system outputs a dollar price estimate. It covers the complete ML lifecycle: data engineering, classical deep learning, LLM fine-tuning, and rigorous evaluation.

**Goal:** Predict USD retail price from product description text with minimal Mean Absolute Error (MAE) and high R² score.

## 🎯 Problem Statement

This is a supervised regression problem. The input is unstructured product text; the output is a continuous price in USD. The challenge is significant because prices span three orders of magnitude ($0.50 to $999) and are influenced by subtle cues like material quality, brand prestige, and feature density.

## 🏗️ Three Approaches Compared


* Baseline : Deep Neural Network ; HashingVectorizer + Residual blocks (PyTorch) 
* Open-source LLM : QLoRA Fine-tuning ; Llama 3.2-3B + 4-bit NF4 quantization + LoRA adapters 
* Commercial API : Gemini Fine-tuning ; Gemini 2.5 Flash via Google AI Studio
  
## 📊 Dataset

- **Source:** McAuley-Lab/Amazon-Reviews-2023 on HuggingFace Hub
- **Content:** Product metadata (title, description, features, price) across 20+ Amazon categories
- **Price range:** $0.50 to $999.49 (outliers filtered)
- **Text length filter:** Minimum 600 characters per product to ensure sufficient context

## ⚙️ Data Pipeline

The system runs a 6-stage pipeline:

1. **Data Ingestion** — Load raw records from HuggingFace Hub using multiprocessing
2. **Parsing & Filtering** — Clean HTML, apply filters, extract fields into structured Item objects
3. **LLM Preprocessing** — Summarize each product with a fast LLM into a standardized 5-field format (Title, Category, Brand, Description, Details)
4. **Dataset Preparation** — Build prompt templates, train/val/test split, push to HuggingFace Hub
5. **Model Training** — Train three parallel tracks: DNN, QLoRA fine-tuning, Gemini API fine-tuning
6. **Evaluation** — Measure MAE, MSE, R² on a held-out test set; visualize with Plotly scatter and trend charts

## 🧠 Model 1 — Deep Neural Network Baseline

- **Feature engineering:** HashingVectorizer with 5,000 features, binary encoding, English stop words removed
- **Price normalization:** Log-transform then standardize to N(0,1) — critical because raw prices span 2000x range
- **Architecture:** Input (5,000-dim) → Linear → LayerNorm → ReLU → Dropout → 8× Residual Blocks → Scalar output
- **Each residual block:** Linear(4096→4096) + LayerNorm + ReLU + Dropout(0.2) + skip connection
- **Skip connections** prevent vanishing gradients in deep networks
- **Training:** AdamW optimizer, L1Loss, CosineAnnealingLR scheduler, batch size 64, gradient clipping

## 🦙 Model 2 — QLoRA Fine-tuning (Llama 3.2-3B)

### Why QLoRA?
Full fine-tuning of Llama 3.2-3B (3 billion parameters) requires 18GB+ GPU memory. The free Colab T4 has 16GB. QLoRA solves this with two techniques:

- **4-bit NF4 quantization:** Stores weights as 4-bit integers → ~1.5 GB (vs 6 GB in fp16), reducing memory by 4×
- **LoRA adapters:** Adds small trainable low-rank matrices (A and B) alongside frozen base weights. Only ~18.4 million parameters are trained instead of 3 billion — a 163× reduction

### LoRA Details
- **Rank:** r=32 (lite, T4 GPU) or r=256 (full, A100 GPU)
- **Target modules:** q_proj, k_proj, v_proj, o_proj (attention layers)
- **Scaling:** lora_alpha=64, so effective scaling = alpha/r = 2.0
- **Trainable params:** ~18.4M across 28 transformer layers
### Training Format
The model is trained as a text completion task:
```
What does this cost to the nearest dollar?
[product summary]
Price is $[amount].00
```
At inference, the prompt is cut off at "Price is $" and the model generates the price.
### Training Config
- **Trainer:** SFTTrainer from TRL library
- **Optimizer:** paged_adamw_32bit (offloads to CPU RAM)
- **LR schedule:** Cosine with 1% warmup
- **Experiment tracking:** Weights & Biases
- **Storage:** HuggingFace Hub (private)
## ✨ Model 3 — Gemini 2.5 Flash Fine-tuning
- Fine-tuned via Google AI Studio API — fully managed, no GPU required
- Training data uploaded as JSONL with conversational format `{messages: [{role, content}]}`
- Same prompt template as QLoRA approach for fair comparison
- Deployed and served via Google API endpoint
## 📈 Evaluation
- **MAE (Mean Absolute Error):** Average dollar error — most intuitive metric
- **MSE (Mean Squared Error):** Penalizes large errors quadratically
- **R² (Coefficient of Determination):** % of price variance explained (0% = always predict mean, 100% = perfect)
- **Color coding:** Green = error < $40 or < 20%, Orange = error < $80 or < 40%, Red = ≥ $80 and ≥ 40%
- **Charts:** Plotly scatter (predicted vs actual) and cumulative error trend with 95% CI bands
- **Parallelized:** ThreadPoolExecutor with 5 workers for fast LLM-based evaluation
## ⚙️ Tech Stack
- **Deep Learning:** PyTorch, scikit-learn
- **LLM Fine-tuning:** HuggingFace Transformers, PEFT (LoRA), TRL (SFTTrainer), BitsAndBytes (quantization)
- **Data:** HuggingFace Datasets, Pydantic (data model)
- **LLM Preprocessing:** LiteLLM (multi-provider), Ollama (local Qwen3:4b)
- **Evaluation:** Plotly, scikit-learn metrics
- **Tracking:** Weights & Biases
- **Training hardware:** Google Colab T4 (16GB) and A100 (40GB) GPUs
## 🔑 Key Technical Concepts
- **ProcessPoolExecutor for data loading** — CPU-bound string parsing bypasses Python's GIL
- **ThreadPoolExecutor for API calls** — I/O-bound LLM calls keep many requests in flight simultaneously
- **Log price normalization** — prevents high-price items from dominating gradients during training
- **Cosine annealing LR** — high LR for exploration early, decays to zero for precise convergence
- **Regression via text generation** — LLM predicts price as a text string "Price is $X.00", extracted via regex post-processing
## 📊 Key Numbers
* Base model : Llama 3.2-3B (3 billion parameters) 
* Memory after 4-bit quantization : ~1.5 GB (vs 6 GB in fp16) 
* LoRA trainable parameters (lite) : ~18.4 million (163× fewer) 
* Transformer layers in Llama 3.2-3B : 28 
* Hidden dimension : 3072 
* Price range in training data : $0.50 to $999.49 
* DNN input features : 5,000 (HashingVectorizer) 
* DNN residual blocks : 8 