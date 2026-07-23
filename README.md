# ⚖️ AI for Legal Case Outcome Prediction

> An end-to-end AI-powered Legal Decision Support System that predicts appellate outcomes from Indian court judgments using fine-tuned **InLegalBERT** transformer models, with integrated **IPC-to-BNS statutory translation** and a secure, role-based **Streamlit** dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-InLegalBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/law-ai/InLegalBERT)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Modules Explained](#-modules-explained)
- [AI Models & Performance](#-ai-models--performance)
- [IPC to BNS Mapping](#-ipc-to-bns-mapping)
- [Database Design](#-database-design)
- [Role-Based Access Control](#-role-based-access-control)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [Team & Acknowledgments](#-team--acknowledgments)
- [References](#-references)

---

## 🔍 Problem Statement

The Indian judiciary processes **5+ crore pending cases** across 25+ High Courts and 700+ District Courts. Lawyers and litigants spend months awaiting outcomes with no predictive insight. The challenge is:

- Court judgments are **lengthy, unstructured PDF documents** (20–100+ pages)
- No standardized AI tools exist for the **Indian legal domain**
- The 2023 transition from **Indian Penal Code (IPC)** to **Bharatiya Nyaya Sanhita (BNS)** creates confusion among legal professionals
- Generic LLMs (GPT, LLaMA) lack domain-specific understanding of Indian statutes

This project addresses these gaps with a purpose-built, domain-specific AI platform.

---

## 🎯 Objectives

1. **Predict case outcomes** (Allowed/Dismissed) from raw court judgment PDFs using deep learning
2. **Classify jurisdiction** (Civil / Criminal / Constitutional) automatically
3. **Translate legacy IPC sections** to their BNS equivalents in real time
4. **Provide role-based access** (Student, Lawyer, Judge, Admin) with secure authentication
5. **Store and track** prediction history in a cloud database for audit purposes

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 📄 **Automated PDF Parsing** | PyMuPDF extracts text; custom regex strips headers, metadata, and legal boilerplate |
| 🧠 **Dual-Inference Pipeline** | Module B predicts jurisdiction (3 classes); Module C predicts outcome (2 classes) |
| ⚖️ **Domain-Specific NLP** | InLegalBERT pre-trained on 5M+ Indian court documents — outperforms generic LLMs |
| 🔄 **IPC → BNS Translation** | 80+ section mappings for real-time statutory code conversion |
| 🔐 **Secure Authentication** | Supabase Auth with role-based access control (RBAC) |
| 📊 **Prediction History** | Cloud-stored audit logs with confidence scores and timestamps |
| 💬 **AI Legal Assistant** | Conversational interface for IPC/BNS queries |
| 📚 **Research Vault** | Private repository of analyzed cases with outcome metadata |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      USER INTERFACE                       │
│               Streamlit Web Dashboard (app.py)            │
│    ┌──────────┬───────────┬──────────┬──────────────┐    │
│    │  Login   │ Predictor │  Vault   │  IPC-BNS Lab │    │
│    │  Portal  │  Module   │  Module  │    Module     │    │
│    └──────────┴───────────┴──────────┴──────────────┘    │
└────────────────────────┬─────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌────────────────┐
│  PDF Parser │  │  InLegalBERT │  │  BNS Mapping   │
│  (PyMuPDF)  │  │  Fine-Tuned  │  │  (JSON DB)     │
│  + Regex    │  │  Models      │  │  80+ Sections  │
│  Cleaner    │  │  Module B/C  │  │                │
└─────────────┘  └──────────────┘  └────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Supabase Cloud    │
              │   PostgreSQL DB     │
              │  ┌───────────────┐  │
              │  │ case_predict. │  │
              │  │ cases_vault   │  │
              │  │ student_query │  │
              │  │ profiles      │  │
              │  └───────────────┘  │
              └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.32+ | Interactive web dashboard |
| **AI/ML** | PyTorch 2.2+, Transformers 4.38+ | Deep learning inference |
| **NLP Model** | InLegalBERT (law-ai) | Domain-specific Indian legal language model |
| **PDF Processing** | PyMuPDF (fitz) | Text extraction from court judgments |
| **NLP Pipeline** | spaCy 3.7+ | Entity recognition and text preprocessing |
| **Database** | Supabase (PostgreSQL) | Cloud authentication + data storage |
| **Auth** | Supabase Auth | Email/password authentication with RBAC |
| **Language** | Python 3.10+ | Backend logic |

---

## 📁 Project Structure

```
Realistic_LJP/
├── .streamlit/
│   ├── config.toml              # Streamlit theme configuration
│   └── secrets.toml             # Supabase connection credentials
├── models/
│   ├── Module_B/Final/          # Jurisdiction classifier (Civil/Criminal/Constitutional)
│   │   ├── config.json
│   │   ├── model.safetensors    # ~438 MB fine-tuned weights
│   │   ├── tokenizer.json
│   │   └── training_args.bin
│   └── Module_C/Final/          # Outcome predictor (Allowed/Dismissed)
│       ├── config.json
│       ├── model.safetensors    # ~438 MB fine-tuned weights
│       ├── tokenizer.json
│       └── training_args.bin
├── app.py                       # Main application (911 lines)
├── bns_mapping.json             # IPC → BNS section mapping (80+ entries)
├── migration.sql                # Database schema + RLS policies
├── requirements.txt             # Python dependencies
├── avatar.jpg                   # User avatar asset
└── README.md                    # This file
```

---

## 🧩 Modules Explained

### Module 1: Authentication & Role-Based Access
- Email/password login via **Supabase Auth**
- Four roles: **Student**, **Lawyer**, **Judge**, **Admin**
- Each role sees only their authorized modules
- Profiles table auto-created on signup via PostgreSQL trigger

### Module 2: Case Predictor (AI Engine)
- Upload PDF/TXT court judgments
- **Text Preprocessing Pipeline:**
  1. PDF → Raw text (PyMuPDF)
  2. Strip headers, coram, metadata (Regex)
  3. IPC → BNS translation (JSON mapping)
  4. Structure-aware chunking (Facts / Arguments / Judgment)
- **Dual Model Inference:**
  - Module B → Jurisdiction classification (3 classes)
  - Module C → Outcome prediction (2 classes)
- Results: Category, confidence %, outcome, progress bar visualization

### Module 3: Research Vault
- Save analyzed cases to Supabase cloud
- Browse historical cases in a card grid
- Each card shows: title, category, outcome, confidence, date

### Module 4: IPC-BNS Statutory Lab
- ChatGPT-style conversational interface
- Input any IPC section → get BNS equivalent instantly
- 80+ statutory mappings from `bns_mapping.json`
- Conversation history stored in database

### Module 5: Case History (Audit Logs)
- Full tabular log of all predictions made
- Columns: filename, jurisdiction, confidence, outcome, date
- Filterable and sortable via Streamlit dataframe

---

## 🤖 AI Models & Performance

### Base Model: InLegalBERT
- **Source:** [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) (IIT Kharagpur)
- **Pre-trained on:** 5 million+ Indian court documents
- **Architecture:** BERT-base (12 layers, 768 hidden, 12 attention heads)
- **Tokenizer:** WordPiece, max sequence length = 512 tokens

### Fine-Tuning Details

| Parameter | Module B (Jurisdiction) | Module C (Outcome) |
|---|---|---|
| **Task** | Multi-class classification | Binary classification |
| **Classes** | Civil, Criminal, Constitutional | Allowed/Accepted, Dismissed/Rejected |
| **Model Size** | ~438 MB | ~438 MB |
| **Format** | SafeTensors | SafeTensors |

### Performance Metrics

| Metric | Score |
|---|---|
| **Accuracy** | 85.2% |
| **Precision** | 84.1% |
| **F1-Score** | 83.7% |
| **Baseline (TF-IDF + SVM)** | 62.0% |

> **Improvement over baseline: +23.2% accuracy**

---

## 🔄 IPC to BNS Mapping

With India's transition from the **Indian Penal Code (1860)** to the **Bharatiya Nyaya Sanhita (2023)**, our system provides instant cross-referencing:

| IPC Section | BNS Section | Offence |
|---|---|---|
| Section 302 | Section 103 of BNS | Punishment for Murder |
| Section 307 | Section 109 of BNS | Attempt to Murder |
| Section 376 | Section 64 of BNS | Punishment for Rape |
| Section 420 | Section 316 of BNS | Cheating & Dishonesty |
| Section 498A | Section 85 of BNS | Cruelty by Husband/Relatives |
| Section 302 | Section 103 of BNS | Murder |

**Total mappings: 80+ sections** stored in `bns_mapping.json`

---

## 🗄️ Database Design

**Platform:** Supabase (hosted PostgreSQL)

### Tables

| Table | Purpose | Key Columns |
|---|---|---|
| `case_predictions` | AI prediction audit log | user_email, filename, jurisdiction, outcome, confidence |
| `cases_vault` | Saved research cases | title, summary, predicted_outcome, case_category |
| `student_queries` | IPC-BNS chat history | user_email, query_text, ai_response |
| `profiles` | User role management | user_id, email, role |
| `statutory_bridge` | IPC-BNS mapping metadata | ipc_section, bns_equivalent |

### Security
- **Row Level Security (RLS)** enabled on all tables
- Auto-profile creation via PostgreSQL trigger on user signup
- Performance indexes on user_email, case_category, ipc_section

---

## 🔐 Role-Based Access Control

| Role | Accessible Modules |
|---|---|
| **Student** | IPC-BNS Lab |
| **Lawyer** | Case Predictor, Research Vault |
| **Judge** | Case History |
| **Admin** | All Modules |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nikhil270703/AI-For-legal-case-outcome-prediction.git
cd AI-For-legal-case-outcome-prediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 4: Configure Supabase
Create `.streamlit/secrets.toml`:
```toml
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-anon-key"
```

### Step 5: Run Database Migration
Execute `migration.sql` in the Supabase SQL Editor to create tables.

### Step 6: Launch the Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📖 Usage Guide

1. **Login** — Enter email/password and select your role
2. **Upload** — Drag and drop a court judgment PDF into the Case Predictor
3. **Analyze** — The system processes the document through the NLP pipeline
4. **Review** — View jurisdiction classification, outcome prediction, and confidence scores
5. **Save** — Archive results to the Research Vault for future reference
6. **Translate** — Use the IPC-BNS Lab to convert legacy section numbers

---

## 🔮 Future Scope

- **Multilingual Support** — Hindi, Marathi, Tamil court documents
- **Sentence Length Prediction** — Predict imprisonment duration
- **Settlement Amount Estimation** — For civil and consumer cases
- **SHAP Explainability** — Visual heatmaps showing which words influenced the prediction
- **API Integration** — RESTful API for third-party legal-tech platforms
- **Mobile Application** — React Native cross-platform app

---

## 👨‍💻 Team & Acknowledgments

| Name | Role |
|---|---|
| **Nikhil Raghuwanshi** | Team Lead, Full-Stack Development, AI Pipeline |
| **Shital Kalekar** | Data Collection, Model Training |
| **Shraddha Dhanwate** | Frontend UI/UX, Testing |
| **Amay Lohar** | Database Design, Documentation |

**Guide:** Prof. Priyanka Patil

**Institution:** Department of CSE (Artificial Intelligence)
Nutan College of Engineering and Research, Pune

---

## 📚 References

1. Paul, S., et al. "InLegalBERT: A Pre-trained Language Model for the Indian Legal Domain." IIT Kharagpur, 2022.
2. Chalkidis, I., et al. "LEGAL-BERT: The Muppets Straight Out of Law School." EMNLP, 2020.
3. Bharatiya Nyaya Sanhita, 2023 — Ministry of Law and Justice, Government of India.
4. Supabase Documentation — https://supabase.com/docs
5. Streamlit Documentation — https://docs.streamlit.io

---

## 📄 License

This project is developed for academic purposes as part of the B.Tech Final Year Project (2024–25).

---

<p align="center">
  <b>Made with ❤️ by Team Legal AI Hub — Nutan College of Engineering, Pune</b>
</p>
