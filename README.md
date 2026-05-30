# 🚀 Real-Time Customer Churn Prediction (MLOps Project)

## 📌 Project Overview
This project is a **real-time end-to-end MLOps pipeline** for predicting customer churn using machine learning, streaming data, and production-grade deployment tools.

It demonstrates how ML models move from training → deployment → real-time inference → monitoring.

---

## 🧠 Key Features
- Real-time data ingestion using **Kafka**
- Machine Learning model for churn prediction
- REST/API or Streamlit-based inference app
- PostgreSQL database for storing predictions
- Monitoring with **Prometheus + Grafana**
- Containerized using **Docker**
- Modular MLOps architecture

---

## 🏗️ Architecture

Producer → Kafka → Consumer → ML Model → PostgreSQL → Dashboard (Grafana)

---

## ⚙️ Tech Stack

- Python
- Scikit-learn / Pandas / NumPy
- Kafka
- PostgreSQL
- Streamlit / Flask
- Docker & Docker Compose
- Prometheus & Grafana

---

## 📁 Project Structure
Customer Churn Prediction/
│
├── producer/
├── consumer/
├── models/
├── app/
├── monitoring/
├── docker-compose.yml
└── README.md


---

## 🚀 How to Run

### 1. Clone Repo
```bash
git clone https://github.com/sandeshkadam-stack/real-time-customer-churn-prediction.git
cd real-time-customer-churn-prediction
docker-compose up --build
streamlit run app.py

📊 Monitoring
Prometheus: metrics collection
Grafana: dashboards for real-time system monitoring

🎯 Use Case
Predict customer churn in telecom/banking systems
Real-time analytics for customer behavior
Production ML pipeline demonstration

📌 Author

Sandesh Kadam
MLOps / Data Science Enthusiast