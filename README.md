## DistilBERT-based Emotion Recognition for Autism Communication 
*A transformer-powered NLP approach to classify emotion expression in ASD specific texts*

---

## Overiew 
- Traditional emotion recognition models fall short when applied to Autism Spectrum Disorder communication due to its unique linguistic and emotional patterns.
- This project fine-tunes a DistilBERT model to identify emotions in text from individuals with ASD,
- aiming to bridge this gap and support better understanding and personalization in educational, therapeutic, and social settings.

---

## Objectives
- Fine-tune DistilBERT for emoition recognition tailored to ASD-related communication.
- Classify text into 5 core emotions: **Joy**, **Sadness**, **Anger**, **Anxiety**, and **Neutral**.
- Ensure interpretability using model explanation techniques( LIME).
- Apply the model to real-world communication scenarios.

---

## Dataset
- **Source**: GoEmoitions by Google Research
- Total size: 58,000+ English Reddit comments
- Fitered for 5 emotions: joy, sadness, anger, anxiety(fear), neutral
- Data type: Text comments + multi-label annotations
- Challenges: Class imbalance, nuance in neutral/anxious tone, ASD-specific variation

---
## Methodology 

**Exploratory Data Analysis**
- Visualized class imbalance
- Analyzed distribution of emotion labels and sample text structure
  
**Preprocessing**
- Filtered to 5 emotion categories
- Handled label encoding and text cleaning
- Split into train/val/test
  
**Model Training**
- Used pre-trained DistilBERT via Hugging Face
- Fine-tuned for multi-class classification
- Used cross-entropy loss and early stopping
  
**Model Interpretability**
- Applied LIME for local explanation of predictions
- Used attention weights to visualize contribution of key tokens

---

## Results 
- **Accuracy**: 87.7%  
- **F1 Score**: 0.88 (macro-averaged)  
- Strong performance in **neutral** and **fear** classification — both common in ASD expression  
- Visual validation of predictions supports model transparency  

---

## Future Work
- Apply model to real ASD support forums and conversations  
- Extend to multi-label settings using Sigmoid output  
- Explore comparison with lexicon-based methods (e.g., VADER)
  
---
## Tools Used
- Python, Jupyter Notebook  
- Hugging Face Transformers (DistilBERT)  
- Scikit-learn, NLTK, Pandas  
- LIME for explainability  
- Matplotlib, Seaborn for EDA
  
