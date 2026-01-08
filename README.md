---

```md
# House Price Prediction in Balikpapan 🏠📈

A **Machine Learning–based web application** for predicting house prices in **Balikpapan City, Indonesia**, using property attributes such as land area, building area, number of rooms, and location (district).

The main model used is **Random Forest Regressor** with **log-transformed target** and **feature engineering**, which achieved excellent predictive performance.

---

## 🚀 Key Features

- House price prediction using Machine Learning
- Property input parameters:
  - Land Area (m²)
  - Building Area (m²)
  - Number of Bedrooms
  - Number of Bathrooms
  - District in Balikpapan
- Output:
  - Predicted house price (IDR)
  - Estimated price range (± MAE)
- Simple web interface built with **Flask + HTML (Jinja2)**

---

## 🧠 Machine Learning Model

### Best Performing Model
- **Random Forest Regressor**
- Target Variable: `log(House Price)`
- Feature Engineering:
  - `price_per_m2_land`
  - `building_to_land_ratio`

### Model Performance (Test Set)
| Metric | Value |
|------|------|
| R² Score | **0.969** |
| MAE | **IDR 117,555,135** |
| RMSE | **IDR 209,826,935** |

The model shows **high accuracy**, **low error**, and **minimal overfitting**.

---

## 📁 Project Structure

```

PREDIKSI-HARGA-RUMAH-BPN
│
├── app.py                     # Flask application (inference)
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html             # Web interface
│
├── model/
│   ├── model_random_forest_harga_rumah.pkl
│   ├── kolom_fitur_model1.pkl
│   └── model_regresi_linear_harga_rumah.pkl
│
├── data_final_bersih.csv      # Cleaned dataset
├── model_training.ipynb       # Model training notebook
└── data_collection.ipynb      # Data preprocessing & exploration

````

---

## ⚙️ How to Run the Application

### 1️⃣ Create Virtual Environment (Optional)
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

Open in your browser:

```
http://127.0.0.1:5001
```

---

## 📝 Important Notes

* The model is saved using `joblib` in `.pkl` format
* A warning may appear if the `scikit-learn` version used during training differs from runtime; this **does not affect prediction results**
* The MAE constant used to display price range in the app:

```python
MAE_FINAL = 473_813_412
```

This value is used to generate **estimated minimum and maximum prices**.

---

## 🎓 Academic Context

This project was developed for:

* **Introduction to Artificial Intelligence** course
* Case study on house price prediction
* Academic presentation / campus expo

Main focus areas:

* Data cleaning and preprocessing
* Feature engineering
* Regression model comparison
* Deployment of ML model into a web application

---

## 👤 Author

**Muhammad Azka Yunastio**
Informatics Engineering Student
Institut Teknologi Kalimantan

---

## 📌 License

This project is intended for **educational and academic purposes only**.

```

---