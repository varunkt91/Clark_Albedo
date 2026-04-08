# 🌍 Albedo Modeling using Sentinel-2 & MODIS  
### Random Forest & Neural Network Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Overview

This project builds a machine learning pipeline to estimate **surface albedo** using:

- 🛰️ Sentinel-2 features (spectral bands, indices)
- 🌍 MODIS-derived albedo (target variable)

The workflow includes:
- Data preprocessing & splitting  
- Model training (Random Forest + Neural Network)  
- Hyperparameter tuning  
- Model evaluation  
- Land cover–specific analysis  

---

## 📂 Project Structure

```
├── Data/
│   ├── raw/
│   ├── processed/
│   └── Data_split/
│
├── models/
│
├── outputs/
│   ├── plots/
│   └── metrics/
│
├── src/
│   ├── data_split.py
│   ├── visualization.py
│   ├── accuracy_matrix.py
│   ├── randomforest_config.py
│
├── notebooks/
│   └── main.ipynb
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/albedo-model.git
cd albedo-model
```

### 2. Create environment

```bash
conda create -n albedo python=3.9
conda activate albedo
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Data Description

Input CSV should contain:

| Feature Type | Description |
|-------------|------------|
| X1, X2...   | Sentinel-2 bands & indices |
| topo_*      | Topographic variables |
| y           | MODIS Albedo |

---

## 🔀 Data Splitting

Dataset is split into:

- **Train** → 40%  
- **Validation** → 30%  
- **Test** → 30%  

```python
split_csv(
    input_path="data.csv",
    output_dir="Data_split/",
    train_ratio=0.4,
    val_ratio=0.3,
    test_ratio=0.3,
    random_state=42
)
```

---

## 🤖 Model Training

### 🌲 Random Forest

```python
train_random_forest(...)
```

---

### 🎯 Hyperparameter Tuning

```python
train_random_forest_randomized_train_val(...)
```

---

### 🔁 Cross Validation

```python
cross_validate_rf(...)
```

---

### ✅ Final Model

```python
train_rf_final(...)
```

---

## 📈 Model Evaluation

Metrics used:

- **R²**
- **RMSE**
- **MAE**

```python
regression_metrics(...)
compute_metrics(...)
```

---

## 📊 Visualization

### 📉 Global Performance

```python
plot_predicted_vs_actual(...)
plot_predicted_vs_actual_density(...)
```

---

### 🌾 Land Cover–wise Analysis

```python
plot_and_save_land_cover_metrics(...)
plot_scatter_per_land_cover_with_metrics(...)
plot_density_scatter_per_land_cover(...)
```

---

### 🌈 Feature Importance

```python
plot_feature_importance(...)
```

---

## 🧠 Neural Network (Optional)

```python
plot_predicted_vs_actual_NN(...)
```

---

## 🗺️ Land Cover Classes

| Class | Description |
|------|------------|
| 1    | Water |
| 2    | Flooded Rice |
| 3    | Other |

---

## 🚀 Workflow Summary

Data → Split → Train → Tune → Validate → Test → Analyze → Visualize

---

## ⚡ Tips

- Use subset for testing  
- Use full dataset for final model  
- Validate across land cover types  
- Check feature importance  

---

## 👨‍💻 Author

Varun Tiwari  

---

## 📜 License

MIT License
