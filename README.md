# 🌱 Smart Farm Sprinkler Predictor  
*Smart Automated Irrigation Using Soil Moisture and Weather Data*

This project implements an AI/ML-based smart irrigation system that predicts which farm sprinklers should be turned **ON/OFF** using real-time sensor data. The system automates irrigation decisions for three farm parcels and exposes an interactive web app built with **Streamlit**.

> Developed as part of the AICTE – Edunet / IBM SkillsBuild Internship (Smart Irrigation Project).

---

## Objectives

- Apply **AI and Machine Learning** techniques to design a **smart, automated irrigation system** for real-world agricultural decision-making.  
- Perform **data cleaning, preprocessing, and feature scaling** using `MinMaxScaler` to prepare sensor data for model training.  
- Build and evaluate a **multi-output Random Forest classification model** using performance metrics such as **precision, recall, and F1-score**.  
- Develop an **interactive Streamlit web application** to predict sprinkler ON/OFF status using real-time sensor inputs.  
- Visualize **sprinkler behavior and pump activity** using `matplotlib`, and **save/load trained models with joblib** for reusable deployment.  


---

## Problem Statement

Traditional irrigation often relies on fixed schedules or manual judgment, which can cause **over-irrigation** (water wastage, root rot) or **under-irrigation** (stress and poor yield).

This project aims to:

- Use **20 sensor readings** (e.g., temperature, humidity, soil moisture, etc.)  
- Predict **3 binary labels**: `parcel_0`, `parcel_1`, `parcel_2` (sprinkler ON/OFF)  
- Automate sprinkler control to ensure efficient and timely irrigation for each parcel.

---

## Machine Learning Approach

### 1. Data Collection & Inspection
- Input: Dataset with **20 sensor features** and **3 target columns** (`parcel_0`, `parcel_1`, `parcel_2`).  
- Performed:
  - Shape and basic statistics
  - Null/missing value checks
  - Distribution understanding of features and labels.

### 2. Data Preprocessing
- Dropped unnecessary columns like `Unnamed: 0`.  
- Applied **MinMaxScaler** to scale sensor values between **0 and 1**.  
- Split the data into:
  - **80% Training**
  - **20% Testing**

### 3. Model Selection & Training
- Base model: **RandomForestClassifier** (robust for tabular data).  
- Wrapped inside **MultiOutputClassifier** to predict all three parcels simultaneously.  
- Tuned hyperparameters such as:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`  
- Trained on scaled training data.

### 4. Model Evaluation
- Evaluated on the test set using **classification reports**:
  - Precision  
  - Recall  
  - F1-score for each parcel.  
- Visualized:
  - **Sprinkler ON/OFF patterns** over time.  
  - **Pump activity** and combined coverage using step plots.

### 5. Model Saving
- Saved the trained model and scaler using **joblib** into:
  - `Farm_Irrigation_System.pkl` (model)
  - `scaler.pkl` (if saved separately)

---

## Tech Stack

**Programming Language**
- Python 3.12.0  

**Development Environment**
- VS Code  
- Streamlit

**Machine Learning**
- `RandomForestClassifier`  
- `MultiOutputClassifier`  

**Libraries**
- `pandas` – data handling  
- `numpy` – numerical operations  
- `matplotlib` – visualizations  
- `scikit-learn` – ML models & preprocessing  
- `joblib` – model persistence  
- `streamlit` – web interface  

---

## Key Features

1. **Smart Decision Making with ML**  
   - Predicts whether sprinklers for `parcel_0`, `parcel_1`, and `parcel_2` should be **ON** or **OFF** based on real sensor readings.

2. **Prevention of Over/Under Irrigation**  
   - Uses combined information from soil moisture, humidity, temperature, etc., to recommend irrigation only when needed.

3. **Automated Sprinkler Control**  
   - Enables full automation instead of manual switching, reducing human effort and response time.

4. **Optimized Water Usage**  
   - Activates only the required sprinklers, supporting **sustainable water management**.

5. **Interactive Web App**  
   - User-friendly **Streamlit** app with sliders for all **20 input sensors**.  
   - Displays predicted sprinkler status in a clear ON/OFF format.

6. **Insightful Visualizations**  
   - Sprinkler activity over time.  
   - Pump activity and farm coverage plots for operational understanding.

---

## Project Structure (Suggested)

```bash
smart-farm-sprinkler-predictor/
├── data/
│   └── sensors_dataset.csv
├── models/
│   ├── Farm_Irrigation_System.pkl
│   └── scaler.pkl
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── model_training.ipynb
├── images/
│   ├── ui_dashboard.png
│   ├── sprinkler_patterns.png
│   └── pump_activity.png
├── requirements.txt
└── README.md

```
---

##  How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/smart-farm-sprinkler-predictor.git
cd smart-farm-sprinkler-predictor
```

### 2. Create & Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (If Needed)

* Open `notebooks/model_training.ipynb` or your training script.
* Run all cells to:

  * Load data
  * Preprocess & scale
  * Train the Random Forest + MultiOutputClassifier
  * Save model & scaler into `models/`.

### 5. Run the Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

* Open the URL shown in the terminal (default: `http://localhost:8501`).
* Adjust the **20 sensor sliders** and click **Predict** to view sprinkler ON/OFF status.

---

##  Results & Insights

* The Random Forest–based multi-output model achieves strong performance across all three parcels (high precision, recall, and F1-score).
* Visual analysis of sprinkler and pump activity confirms that:

  * Irrigation is triggered when sensor conditions demand it.
  * Pump and sprinkler usage is more optimized compared to fixed-time schedules.
* The system demonstrates how **data-driven irrigation** can reduce manual effort and water wastage.

---

##  Future Scope

1. **IoT Integration**

   * Connect physical soil moisture, temperature, and humidity sensors directly to the model for real-time updates.

2. **Weather Forecast Integration**

   * Use weather APIs to reduce irrigation when rain is expected or increase it during heat waves.

3. **Mobile App Dashboard**

   * Create a mobile app for farmers to monitor and control irrigation remotely.

4. **Scalability for Large Farms**

   * Extend the system to handle more parcels/crop zones and different crop requirements.

5. **Energy Optimization**

   * Integrate solar-powered pumps and smart scheduling to reduce electricity consumption.

6. **Alerts & Reporting**

   * Add SMS or app notifications for pump failures, abnormal sensor values, or missed irrigation cycles.
   * Generate monthly water usage and irrigation effectiveness reports.

---

