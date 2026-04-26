# 🏷️ The Price Is Right — AI Product Price Predictor

An end-to-end ML system that predicts the retail price of a product from its text description, trained on the Amazon Reviews 2023 dataset.

---

## What it does

Given a product's title, category, brand, and feature details, the system outputs a dollar price estimate. Three model approaches are compared against a deep learning baseline.

---

## Approaches

| Model | Method | Hardware |
|-------|--------|----------|
| Deep Neural Network | HashingVectorizer + Residual blocks | CPU / any GPU |
| Llama 3.2-3B (QLoRA) | 4-bit quantization + LoRA adapters | Google Colab T4 / A100 |
| Gemini 2.5 Flash | Managed API fine-tuning | Google AI Studio |
| Qwen3:4b | Local inference via Ollama | 16GB RAM |

---

## Project Structure

```
├── pricer/
│   ├── items.py          # Item data model (Pydantic) + prompt templating
│   ├── loaders.py        # Parallel dataset loading (ProcessPoolExecutor)
│   ├── parser.py         # Filtering, text cleaning, regex scrubbing
│   ├── preprocessor.py   # Single-item LLM summarization via LiteLLM
│   ├── batch.py          # Batch LLM summarization (ThreadPoolExecutor)
│   ├── deep_neural_network.py  # DNN architecture, training loop, inference
│   └── evaluator.py      # MAE / MSE / R² metrics + Plotly charts
│
├── notebooks/
│   ├── day1–4.ipynb                    # Data pipeline + DNN training
│   ├── gemini_fine_tuning.ipynb        # Gemini API fine-tuning
│   ├── NEW_Week7_Day1_qlora_intro.ipynb  # LoRA/QLoRA concepts
│   ├── NEW_Week7_Day3_TRAINING.ipynb   # Full SFTTrainer fine-tuning run
│   └── results.ipynb                   # Model comparison & evaluation
```

---

## Pipeline

```
Amazon Reviews 2023
        ↓
  Parse & Filter        # price $0.50–$999, min 600 chars, strip part numbers
        ↓
  LLM Summarize         # standardize to 5-field format (Title/Category/Brand/Description/Details)
        ↓
  Build Prompts         # "What does this cost?\n\n{summary}\n\nPrice is $XX.00"
        ↓
  Train / Fine-tune     # DNN baseline  |  QLoRA (Llama)  |  Gemini API
        ↓
  Evaluate              # MAE, MSE, R² on held-out test set
```

---

## Key Design Decisions

- **LLM preprocessing** normalizes inconsistent Amazon product formats before fine-tuning
- **ProcessPoolExecutor** for data loading (CPU-bound); **ThreadPoolExecutor** for API calls (I/O-bound)
- **4-bit NF4 quantization** reduces Llama 3.2-3B from ~6GB to ~1.5GB VRAM
- **LoRA (r=32)** cuts trainable parameters from 3B → 18.4M (163× reduction)
- **Log-price normalization** prevents high-price items from dominating DNN gradients
- **Residual connections** enable stable training of deep networks

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| MAE | Average dollar error — most interpretable |
| MSE | Penalizes large errors quadratically |
| R²  | % of price variance explained (0% = predict mean, 100% = perfect) |

---

## Setup

```bash
pip install torch transformers peft trl bitsandbytes datasets litellm pydantic plotly wandb tqdm
```

For local Qwen3 inference, install [Ollama](https://ollama.ai) and run:
```bash
ollama pull qwen3:4b
```

Set environment secrets: `HF_TOKEN`, `WANDB_API_KEY`, and any LLM provider API keys in `.env`.

---

## References

- Dataset: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Base model: [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- LoRA paper: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- QLoRA paper: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)