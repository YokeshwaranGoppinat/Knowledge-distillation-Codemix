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

# 📈 Results Summary

The strongest distilled model achieves performance close to the teacher with **significantly fewer parameters**, making it suitable for deployment on edge devices.


---

# 📝 License
MIT License

