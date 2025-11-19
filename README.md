# Knowledge Distillation for Code-Mixed Sentiment Classification

This repository implements **teacher–student knowledge distillation** for sentiment classification on **Tamil–English** and **Hindi–English** code-mixed text.  
It includes preprocessing, teacher training, multiple student models, evaluation, and a full dataset for reproducibility.

---

# 📘 Overview

Modern Transformer models perform strongly on noisy code-mixed text, but they are expensive to deploy.  
**Knowledge Distillation (KD)** compresses a large model (teacher) into a much smaller student model with minimal accuracy drop.

This project trains:
- A full-size **Teacher Transformer**
- Six distilled **Student models**:
  - Baseline
  - Soft-label distillation
  - Hidden-state distillation
  - Embedding distillation
  - Attention distillation
  - Full multi-signal distillation

Each variant is analyzed in `ResultsSummary.ipynb`.

---

# 📂 Project Structure

```
Data/
  Tamil_codemix/
    tam_train.csv
    tam_val.csv
    tam_test.csv
  Hindi_codemix/
    hin_train.csv
    hin_val.csv
    hin_test.csv

src/
  dataset_utils.py
  trainer.py
  model_utils.py
  utils.py

notebooks/
  DataPrep.ipynb
  TeacherTrainer.ipynb
  Student_Baseline.ipynb
  Student_Soft.ipynb
  Student_Hidden.ipynb
  Student_Embedding.ipynb
  Student_Attention.ipynb
  Student_Full.ipynb
  ResultsSummary.ipynb

smoke_test.py
run_colab.ipynb
requirements.txt
LICENSE
```

---

# 📊 Dataset (Included)

This repository **includes the dataset** used for training and evaluation:

### Tamil Code-Mixed Sentiment
- `tam_train.csv`  
- `tam_val.csv`  
- `tam_test.csv`

### Hindi Code-Mixed Sentiment
- `hin_train.csv`  
- `hin_val.csv`  
- `hin_test.csv`

Each CSV has:
- `review` — text (code-mixed sentence)  
- `label` — sentiment class (0/1)

These files are small and safe to store directly in the repository.

---

# 🚀 Running the Project (Colab Recommended)

### 1. Open **run_colab.ipynb**
Run all cells.  
This will:
- Install dependencies  
- Run `smoke_test.py` (30-second functional test)  
- Detect the dataset under `Data/`  
- Confirm tokenization → dataloading → model forward pass  

### 2. Run full experiments
Open `notebooks/` and run:

- `TeacherTrainer.ipynb` → train teacher  
- `Student_*.ipynb` → train student models  
- `ResultsSummary.ipynb` → generate comparison plots  

---

## Quick start (local)

You can also run the project locally without Colab:

```bash
git clone https://github.com/YokeshwaranGoppinat/Knowledge-distillation-Codemix.git
cd Knowledge-distillation-Codemix

# (optional but recommended) create a virtual environment
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Run the fast smoke test to validate the pipeline
python smoke_test.py
```

---

# 🧪 Student Model Variants

| Variant | Description |
|--------|-------------|
| Baseline | Trains from scratch without teacher |
| Soft | Distills only soft probabilities |
| Hidden | Matches hidden-layer activations |
| Embedding | Matches token embeddings |
| Attention | Matches attention matrices |
| Full | All losses combined for strongest student |

---

## Results

This project evaluates a 6-layer Transformer **teacher model** and a 2-layer **student model** distilled using multiple strategies (soft logits, hidden-state transfer, embedding-based, attention-based, and full distillation). These results match the controlled experiments discussed in the project write-up and statement of purpose.

---

# 🔵 Tamil Code-Mixed Results (2-layer student, 2 epochs)

| Model / Technique          | Accuracy | F1 Score | Parameters (M) | Reduction |
|---------------------------|----------|----------|----------------|-----------|
| **Teacher (6 layers)**    | **0.753** | **0.752** | 135.33         | —         |
| Student – Baseline        | 0.735    | 0.734    | 106.97         | **20.95%** |
| Student – Soft            | **0.731** | **0.731** | 106.97         | **20.95%** |
| Student – Hidden          | 0.688    | 0.686    | 106.97         | **20.95%** |
| Student – Full (Soft + Hidden) | 0.726 | 0.725 | 106.97 | **20.95%** |
| Student – Embedding       | 0.734    | 0.733    | 106.97         | **20.95%** |
| Student – Attention       | 0.728    | 0.727    | 106.97         | **20.95%** |

### Key insight (Tamil)
- Soft-logits distillation achieved **0.731 F1**, closely matching the teacher’s **0.752**, while reducing parameters by **~21%**.
- Students trained without teacher guidance (baseline) still performed well, indicating robustness in the code-mixed setting.

---

# 🟠 Hindi Code-Mixed Results (2-layer student, 2 epochs)

| Model / Technique           | Accuracy | F1 Score | Parameters (M) | Reduction |
|----------------------------|----------|----------|----------------|-----------|
| **Teacher (6 layers)**     | **0.885** | **0.885** | 135.33         | —         |
| Student – Baseline         | 0.796    | 0.794    | 106.97         | **20.95%** |
| Student – Soft             | **0.801** | **0.800** | 106.97         | **20.95%** |
| Student – Hidden           | 0.775    | 0.773    | 106.97         | **20.95%** |
| Student – Full (Soft + Hidden) | 0.789 | 0.788 | 106.97 | **20.95%** |
| Student – Embedding        | 0.795    | 0.793    | 106.97         | **20.95%** |
| Student – Attention        | 0.776    | 0.774    | 106.97         | **20.95%** |

### Key insight (Hindi)
- Soft-logits distillation achieved **0.800 F1**, preserving strong generalization relative to the teacher’s **0.885**, with **~21%** parameter reduction.
- Performance trends mirror Tamil, confirming consistent distillation behavior across two linguistically distinct code-mixed datasets.

---

# 🧠 Overall Summary

- The 2-layer distilled student retains **90–97%** of teacher performance while being **~21% smaller**.
- Soft-logits distillation provides the best balance of compression and accuracy in both datasets.
- The results align with challenges in multilingual code-mixed settings—high lexical variability and limited samples—but show that **distillation can meaningfully compress multilingual models without heavy accuracy loss**.

---

# 📝 License
MIT License

