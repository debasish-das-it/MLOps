# 🚀 MLOps: End-to-End Machine Learning Pipeline

## Project Overview

A comprehensive **Machine Learning Operations (MLOps)** project demonstrating a complete production-ready ML workflow—from data exploration and preprocessing to model deployment. This project showcases enterprise-grade ML practices including data versioning, hyperparameter optimization, model registry, and cloud-based inference.

**Live Demo:** [Tourism Package Predictor App](https://huggingface.co/spaces/debasishdas1985/tourism-package-predictor) on Hugging Face Spaces

---

## 🎯 Key Accomplishments

### 📊 **Data Engineering & Analysis**
- **Comprehensive Exploratory Data Analysis (EDA):** In-depth statistical analysis and visualization of dataset patterns
- **Data Preprocessing Pipeline:** Intelligent data cleaning, transformation, and feature engineering
- **Train-Test Split:** Proper data stratification and splitting for robust model evaluation
- **Dataset Versioning:** All datasets stored and managed as a dedicated repository on Hugging Face Datasets for reproducibility and accessibility

### 🤖 **Model Development & Optimization**
- **Hyperparameter Tuning:** Systematic experimentation with multiple algorithms and parameter configurations
- **Model Comparison:** Rigorous evaluation and comparison of different ML models to identify optimal performance
- **Best Model Selection:** Data-driven approach to selecting the highest-performing model based on multiple metrics
- **Model Registry:** Final model version stored on Hugging Face Hub for version control and easy retrieval

### 🌐 **Deployment & Inference**
- **Automated Data Pipeline:** Seamless download of versioned datasets from Hugging Face at training time
- **Production-Ready Model:** Selected best model packaged and uploaded to Hugging Face for public accessibility
- **Interactive Streamlit Application:** User-friendly web interface for real-time predictions hosted on Hugging Face Spaces
  - URL: https://huggingface.co/spaces/debasishdas1985/tourism-package-predictor

---

## 🏗️ Repository Structure

```
MLOps/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and model evaluation
├── src/                   # Core ML pipeline modules
│   ├── data_processing.py # Data cleaning and feature engineering
│   ├── model_training.py  # Model training and hyperparameter tuning
│   └── inference.py       # Model inference utilities
├── models/                # Trained model artifacts
├── scripts/               # Automation and utility scripts
├── streamlit_app.py       # Interactive Streamlit application for predictions
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/debasish-das-it/MLOps.git
cd MLOps

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python src/model_training.py
```

This script will:
1. Automatically download the dataset from Hugging Face
2. Preprocess and prepare the data
3. Execute hyperparameter tuning across multiple models
4. Train and evaluate candidate models
5. Save the best-performing model

### Running Local Inference

```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to `http://localhost:8501` for interactive predictions.

---

## 📈 Technical Highlights

| Component | Details |
|-----------|---------|
| **Data Source** | Hugging Face Datasets (Tourism Package Dataset) |
| **Model Registry** | Hugging Face Hub |
| **Training Framework** | scikit-learn / TensorFlow / PyTorch |
| **Hyperparameter Tuning** | Grid Search / Random Search / Bayesian Optimization |
| **Deployment Platform** | Hugging Face Spaces (Streamlit) |
| **Version Control** | Git + Hugging Face Hub |

---

## 🔗 Live Resources

- **Streamlit App (Inference):** [Tourism Package Predictor](https://huggingface.co/spaces/debasishdas1985/tourism-package-predictor)
- **Model Hub:** Available on Hugging Face Hub
- **Dataset Repository:** Hosted on Hugging Face Datasets

---

## 💡 Skills Demonstrated

✅ **Machine Learning:** Model building, evaluation, and selection  
✅ **Data Science:** EDA, data preprocessing, feature engineering  
✅ **MLOps:** Data versioning, model registry, CI/CD principles  
✅ **Hyperparameter Optimization:** Systematic tuning and cross-validation  
✅ **Cloud Deployment:** Hugging Face ecosystem integration  
✅ **Web Development:** Streamlit interactive applications  
✅ **Best Practices:** Reproducibility, documentation, version control  

---

## 📝 Contributing

Contributions and suggestions are welcome! Please feel free to:
- Create an issue for bug reports or feature requests
- Submit a pull request with improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Questions or feedback?** Feel free to reach out or open an issue in the repository.
