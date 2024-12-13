

# Transformer Model Comparisons for Non-Autoregressive NLP Tasks  

**New York University | DS-GA-1011: Natural Language Processing with Representation Learning**  

## Overview  

This repository evaluates the performance of different transformer architectures—**encoder-only** (e.g., RoBERTa), **decoder-only** (e.g., GPT-2), and **encoder-decoder** (e.g., T5)—on typically **non-autoregressive** NLP tasks, including:  

- **Named Entity Recognition (NER)**  
- **Text Classification**  
- **Sentence Similarity**  
- **Summarization**  

The experiments analyze models of comparable size (Small, Medium, and Large) using established datasets and metrics for performance evaluation, resource usage, and inference efficiency.  

---

## Folder Structure  

```plaintext
.
├── NER/code for all 3 models                  # Named Entity Recognition code and scripts

│
├── Classification/code for all 3 models       # Text classification task code

│
├── Sentence_Similarity/code for all 3 models   # Sentence similarity code

│
├── Summarisation/code for all 3 models          # Summarization task code

│
└── README.md               # Project description and instructions
```

---

## Requirements  

Ensure you have the following libraries installed:

- Python 3.8+  
- PyTorch  
- Transformers (Hugging Face)  
- NumPy  
- scikit-learn  
- pandas  
- matplotlib  

To install the dependencies, run:  
```bash
pip install torch transformers scikit-learn numpy pandas matplotlib
```

---

## Datasets  

- **Named Entity Recognition (NER):** [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)  
- **Text Classification:** [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
- **Sentence Similarity:** [SICK Dataset](http://clic.cimec.unitn.it/composes/sick.html)  
- **Summarization:** [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)  

---


## Results  

The performance of different architectures on various tasks:  

| Task                | Best Model           | Metric(s) Achieved       |  
|---------------------|----------------------|--------------------------|  
| **NER**            | DeBERTa-XLarge       | Accuracy: 99.24%, F1: 96.57% |  
| **Classification**  | T5-Large             | Accuracy: 95.31%         |  
| **Sentence Similarity** | T5-Base            | Pearson: 0.8854          |  
| **Summarization**   | GPT2-Large           | ROUGE-L: 92.33%          |  

---

## Power Consumption Summary  

- **Decoder-Only Models** (e.g., GPT-2): Most power-hungry (~225–250W).  
- **Encoder-Only Models**: Most power-efficient (~111–150W).  
- **Encoder-Decoder Models**: Balanced power usage and task performance.  

---

## Team Members  

- Dhruv Sridhar  
- Barath Ramashankar  
- Sampreeth Avvari  
- Nihal V P  

---

## Acknowledgments  

This project was conducted as part of **NYU's DS-GA-1011 course** under the supervision of the Center for Data Science.  

---  

## License  

This project is licensed under the MIT License.  

---  

If you use this work or have any questions, feel free to open an issue or contact us!  

---  

