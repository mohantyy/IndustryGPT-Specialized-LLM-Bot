# IndustryGPT-Specialized-LLM-Bot
IndustryGPT: Specialized Enterprise IT Support &amp; Cloud LLM Bot Using Pre-Trained Transformers, 25-Epoch PEFT LoRA, Hybrid FAISS RAG, and LangGraph (AlmaBetter AI Engineering Capstone)
# 🛠️ IndustryGPT: Specialized LLM Bot Using Pre-Trained Models (Enterprise IT Support & Cloud Infrastructure)

**End-to-End Domain-Specific Conversational AI System Powered by Pre-Trained Transformers (Hugging Face), 25-Epoch Low-Rank Adaptation (PEFT LoRA), Hybrid FAISS Dense + BM25 Retrieval-Augmented Generation (RAG), Cross-Encoder Reranking, and LangGraph Stateful Workflow Orchestration**

- **Author:** Abhilash Mohanty (`@mohantyy`)
- **Program:** AlmaBetter Masters in Data Science — AI Engineering v1 (Capstone Project: Deep Learning for NLP)
- **Academic Affiliation:** Master's in CS: Artificial Intelligence and Machine Learning, Woolf University
- **Selected Industry:** Technology and Information Technology (IT) — Enterprise IT Support, Cloud Infrastructure, DevOps, Cybersecurity & ITSM
- **Google Drive Project Folder (Colab Notebook, Dataset, Research Paper & Video Script):** https://drive.google.com/drive/folders/1F2ZG_W9oOfKqF_YfF11kD-1Nu7ti3VJ0
- **Google Colab Notebook Link:** https://colab.research.google.com/drive/1E0Oep386cMyeEXZwbZrAb3A-9te9cuUX

---

## 📌 1. Executive Summary & Business Problem

Enterprise IT service desks and Site Reliability Engineering (SRE) teams manage thousands of recurring technical incidents across cloud clusters, VPN gateways, identity providers, databases, CI/CD pipelines, and endpoint security platforms. **IndustryGPT** bridges this operational gap by combining:
1. **Domain-Specific Data Collection & Wrangling:** A curated corpus of **2,400 enterprise IT support interactions (`2,385` deduplicated records across 16 attributes)** spanning **8 IT domains** and **14 technical runbook sub-categories** (`data/it_support_llm_dataset.csv` and `data/it_support_knowledge_base.jsonl`).
2. **10-Step NLP Preprocessing & Recursive Text Chunking:** Domain-aware normalization preserving technical error tokens (`ERR_K8S_OOM_137`, `ERR_SAML_ASSERTION_401`, `ERR_PG_TOO_MANY_CONN_53300`) paired with `RecursiveCharacterTextSplitter` (`chunk_size=3000, chunk_overlap=500`).
3. **Hybrid FAISS Vector Store & Cross-Encoder Reranker:** Combines 64-dimensional L2-normalized Truncated SVD dense semantic embeddings with Okapi BM25 lexical matching (`top_k=10`) and a Cross-Encoder interaction reranker (`top_k=3`, achieving **97.8% Recall@3**).
4. **25-Epoch PEFT LoRA Fine-Tuning:** Fine-tunes a pre-trained Transformer architecture using Low-Rank Adaptation (`r=16, lora_alpha=32, lora_dropout=0.05`, updating only `1.18%` of parameters) over **25 epochs** with Cosine Annealing learning rate scheduling (`lr=2e-4`), reducing validation cross-entropy loss from `2.9372` (Perplexity `18.86`) to `0.5132` (Perplexity `1.67`).
5. **Stateful LangGraph Conversational Orchestration:** Routes every user prompt through a deterministic `MessagesState` DAG with dedicated nodes for **Ethical & Security Guardrails (`100%` adversarial refusal)**, **Out-of-Domain Fallback (`100%` off-topic containment)**, **Hybrid FAISS Retrieval**, and **Context-Injected LLM Generation**, achieving **95.62% Response Accuracy**, **0.9552 Token F1**, **0.9145 ROUGE-L**, **1.48s Average Latency**, and **4.72 / 5.0 User Satisfaction**.

---

## 📈 2. Model Evaluation Benchmark (Validation Split: `n = 477`)

| Model Architecture | Configuration | Response Accuracy (%) | Precision (%) | Recall (%) | Token F1-Score | ROUGE-L | BLEU-4 | Perplexity | Avg Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Model 1 (Baseline)** | Pre-Trained Zero-Shot LLM + Unigram TF-IDF | 76.42% | 75.18% | 74.65% | 0.7491 | 0.6840 | 0.5420 | 9.84 | 2.45s |
| **Model 1 (5-Fold CV Tuned)** | Okapi BM25 (`k1=1.5, b=0.75`) + Bigram TF-IDF | 81.84% | 81.10% | 80.45% | 0.8077 | 0.7390 | 0.6080 | 7.62 | 2.12s |
| **Model 2 (Baseline)** | Pre-Trained Transformer + LoRA (`r=4`, 10 Epochs) | 86.20% | 85.64% | 84.92% | 0.8528 | 0.8015 | 0.6890 | 2.48 | 1.95s |
| **Model 2 (Tuned)** | Pre-Trained Transformer + LoRA (`r=16, α=32`, 25 Epochs) | 90.58% | 90.12% | 89.65% | 0.8988 | 0.8492 | 0.7540 | 1.67 | 1.78s |
| **Model 3 (Baseline RAG)** | 25-Epoch LoRA LLM + Dense FAISS (`Top-K=5`, No Reranker) | 92.45% | 92.10% | 91.80% | 0.9195 | 0.8760 | 0.7980 | 1.49 | 1.62s |
| **Model 3 (Champion IndustryGPT)** | **25-Epoch LoRA LLM + Hybrid FAISS + Cross-Encoder + LangGraph** | **95.62%** | **95.84%** | **95.21%** | **0.9552** | **0.9145** | **0.8460** | **1.42** | **1.48s** |

---

## 🚀 3. Quickstart & Live Bot Execution

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py" -v
PYTHONPATH=. python3 app.py --cli-demo
streamlit run app.py
```
