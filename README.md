# 🛒 Online Retail Intelligence — Data Mining & Knowledge Discovery

> **INFO411 · University of Wollongong in Dubai**  
> A full-stack data mining pipeline applied to the UCI Online Retail dataset — covering customer segmentation, churn prediction, and revenue forecasting.

[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-blue)](https://archive.ics.uci.edu/ml/datasets/online+retail)
[![Language](https://img.shields.io/badge/Languages-R%20%7C%20Python-informational)](#)
[![Models](https://img.shields.io/badge/Models-KMeans%20%7C%20Decision%20Tree%20%7C%20Neural%20Network%20%7C%20ARIMA-success)](#)
[![Accuracy](https://img.shields.io/badge/Best%20Churn%20Sensitivity-84%25-brightgreen)](#)
[![MAPE](https://img.shields.io/badge/Forecast%20MAPE-3.45%25-brightgreen)](#)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Preprocessing](#%EF%B8%8F-preprocessing)
- [Clustering Analysis](#-clustering-analysis)
- [Classification & Churn Prediction](#-classification--churn-prediction)
- [Time Series Forecasting](#-time-series-forecasting)
- [Key Findings](#-key-findings)
- [Business Recommendations](#-business-recommendations)
- [Team](#-team)

---

## 🔍 Overview

This project applies three core data mining techniques to real-world retail transaction data, transforming over **392,000 raw records** into actionable business intelligence:

| Task | Technique | Goal |
|---|---|---|
| **Customer Segmentation** | K-Means Clustering (RFM) | Identify behavioral groups |
| **Churn Prediction** | Decision Tree + Neural Network | Flag at-risk customers |
| **Revenue Forecasting** | ARIMA Time Series | Predict next 30 days of sales |

---

## 📦 Dataset

- **Source:** UCI Online Retail Dataset
- **Records:** 392,692 transactions
- **Period:** December 2010 – December 2011
- **Customers (post-cleaning):** 4,338 unique customers

### Preprocessing Pipeline

```
Raw Data (541,909 rows)
    │
    ├── Remove missing CustomerIDs    → −135,080 rows (25%)
    ├── Remove cancellations (C*)     → −~8,905 rows
    ├── Remove invalid Qty / Price    → minor removal
    ├── Remove duplicates             → distinct()
    │
    └── Cleaned Dataset (392,692 rows)
            │
            └── Feature Engineering:
                    TotalPrice = Quantity × UnitPrice
                    Year, Month, DayOfWeek (via lubridate)
                    RFM metrics per CustomerID
```

**Notable pattern discovered during EDA:**
- 🔺 **Thursday** is the peak sales day (~80,000 orders)
- 🔻 **Saturday** records zero activity
- 📈 November peaks at **60,000+ orders** — nearly 3× the summer baseline

---

## 🧩 Clustering Analysis

### Methodology
1. **RFM Feature Engineering** — Recency, Frequency, Monetary per customer
2. **Log1p Transformation** — corrects right-skewed distributions
3. **Z-Score Standardization** — equalizes feature contribution
4. **PCA** — 93.86% variance captured in 2 components
5. **K-Means (k=3)** — selected over k=2 for business granularity

### Results

| Cluster | Label | Customers | % | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|---|
| 1 | 🔴 At-Risk / Dormant | 1,871 | 43.1% | 168 days | 1.36 orders | £362 |
| 2 | 🟡 Potential / Regular | 1,704 | 39.3% | 44 days | 3.37 orders | £1,256 |
| 3 | 🟢 Champions | 763 | 17.6% | 17.8 days | 13.4 orders | £7,954 |

> 💡 **Key Insight:** Champions represent only **17.6%** of customers but generate **~3.9×** the overall average revenue per customer.

---

## 🤖 Classification & Churn Prediction

### Churn Definition (RFM-based label)
```
IsChurned = YES  if:
    DaysSinceLastPurchase > 90 days
    OR (PurchaseFrequency ≤ 3 AND TotalSpent ≤ £100)
```

**Distribution:** 65.3% Not Churned · 34.7% Churned

### Features Used
| Feature | Description |
|---|---|
| `ProductDiversity` | # of unique products purchased |
| `AvgBasketValue` | Average spend per order |
| `DaysSinceFirstPurchase` | Customer tenure |
| `AvgQuantity` | Average items per transaction |
| `AvgUnitPrice` | Average price per item |

### Model Comparison

| Model | Accuracy | Sensitivity | Specificity | Precision | Missed Churners |
|---|---|---|---|---|---|
| Baseline Decision Tree (3 features) | 69.6% | 27.4% | 91.3% | 61.7% | 321 |
| Improved Decision Tree (5 features) | 79.7% | 65.2% | 87.2% | 72.4% | 154 |
| **Neural Network (5 features + class weights)** | **80.4%** | **84.0%** | **79.0%** | **68.0%** | **75** |

### Neural Network Architecture
```
Input (5 features)
    → Dense(64, ReLU) + Dropout(0.3)
    → Dense(32, ReLU) + Dropout(0.3)
    → Dense(16, ReLU) + Dropout(0.2)
    → Dense(1, Sigmoid)

Optimizer: Adam | Loss: Binary Cross-Entropy
Epochs: 50 | Batch size: 32 | Class weights: {0:1, 1:2}
```

> 🏆 **Winner: Neural Network** — catches **94 more at-risk customers** than the Decision Tree, at the cost of slightly more false alarms. In a churn context, false negatives are far costlier than false positives.

---

## 📈 Time Series Forecasting

### Pipeline
```
Daily Transactions
    → Aggregate by date (sum TotalPrice)
    → Log transformation (stabilize variance)
    → SMA smoothing (7-day and 30-day windows)
    → STL Decomposition (trend + weekly seasonality)
    → ADF test → stationary (p = 0.01)
    → ARIMA modeling → 30-day forecast
```

### Model Comparison

| Model | MAPE |
|---|---|
| Manual ARIMA (5,1,0) | 4.57% |
| ETS (Exponential Smoothing) | > 4.57% |
| **Auto-ARIMA (5,1,3)(2,0,0)[7]** | **3.45%** ✅ |

> **Auto-ARIMA** achieved the best accuracy by automatically capturing both the autoregressive and moving average patterns revealed in the ACF/PACF analysis, plus the strong weekly seasonal cycle.

### Weekly Seasonality
```
Mon  ████████████░░░
Tue  █████████████░░
Wed  ████████████░░░
Thu  ████████████████  ← Peak day
Fri  ████████████░░░
Sun  ████████░░░░░░░   ← Lowest active day
Sat  ░░░░░░░░░░░░░░░   ← Zero activity
```

---

## 💡 Key Findings

- **Revenue is concentrated:** 17.6% of customers (Champions) generate disproportionate value — losing even 10–15% of them would have a severe revenue impact
- **Feature engineering matters most:** Adding just 2 engagement features (`ProductDiversity`, `AvgBasketValue`) boosted churn sensitivity from **27.4% → 65.2%**
- **Neural networks > Decision Trees for churn:** Especially when class weights are used to reflect real business costs of missed churners
- **Weekly cycles are highly predictable:** 93.86% of variance captured in 2 PCA dimensions; seasonality is strong and exploitable
- **ARIMA forecasts are reliable:** 3.45% MAPE enables dependable 30-day planning

---

## 📋 Business Recommendations

### Customer Strategy
| Segment | Recommended Action |
|---|---|
| 🔴 At-Risk | Win-back email with time-limited discount; low-budget outreach |
| 🟡 Potential | Loyalty program, spend-threshold rewards, cross-sell campaigns |
| 🟢 Champions | VIP tiers, early product access, personalised service |

### Operations (from Time Series)
- **📦 Friday staffing:** Thursday is peak sales day — schedule extra warehouse staff Friday morning for next-day shipping
- **🎯 Sunday flash sales:** Target the weekly revenue dip with app-only promotions
- **📊 Safety stock:** Use the forecast's lower confidence bound to set minimum inventory levels
- **🚀 Product launches:** Release new items Thursday morning to maximise exposure
- **🔄 Returns planning:** Expect customer service spikes on Wednesdays (from Thursday deliveries)

---

## 👩‍💻 Team

| Name | Student ID |
|---|---|
| Marwa Khot | 8963186 |
| Joslin Jolly | 8964178 |
| Zobia Shaikh | 8881820 |

**Lecturer:** Dr. Farhad  
**Subject:** INFO411 — Data Mining and Knowledge Discovery  
**Submitted:** 15 March 2026

---

## 🔗 Repository

📁 [https://github.com/Marwakhot/INFO_411](https://github.com/joslin686/retail-customer-analysis)
