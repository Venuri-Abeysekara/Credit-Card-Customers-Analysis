# Credit Card Customer Churn Analysis

## 📊 Project Overview

This project analyzes credit card customer churn patterns to identify at risk customers and develop data driven retention strategies. Through comprehensive exploratory data analysis and predictive modeling, we built machine learning models that achieve **96%+ AUC accuracy** in identifying customers likely to churn before they leave.

**Analysis Question:** *What factors cause credit card customers to churn and can we identify at risk customers before they leave?*

---

## 🎯 Business Problem

A credit card services manager faces increasing customer attrition, threatening revenue and market share. The organization needs:
- Predictive capabilities to identify at risk customers proactively
- Understanding of key churn drivers
- Actionable strategies for customer retention

---

## 📁 Project Structure

The analysis is divided into three Jupyter notebooks covering the complete data science workflow:

### **1. Basic data set analysis.ipynb**
**Focus:** Initial data exploration and understanding

**Contents:**
- Dataset loading and initial inspection
- Basic statistical summaries
- Data structure examination
- Initial churn rate identification

**Purpose:** First look at the data to understand its characteristics and quality

---

### **2. Exploratory Data Analysis.ipynb**
**Focus:** In depth data exploration and pattern discovery

**Contents:**
- Comprehensive data quality assessment
- Churn distribution analysis (16.1% churn rate)
- Detailed feature correlation analysis
- Statistical comparisons of churners vs. retained customers
- Demographic and behavioral pattern analysis
- Multiple data visualizations (charts, graphs, heatmaps)

**Key Findings:**
- Churned customers average 13 fewer transactions
- Churned customers are inactive 2.2 more months per year
- Strong negative correlation between transaction count and churn (-0.39)
- Months of inactivity shows positive correlation with churn (0.16)
- Transaction activity dominates over demographics in churn prediction

---

### **3. Predictive Modeling & Advanced Analysis.ipynb**
**Focus:** Machine learning models and customer risk segmentation

**Contents:**
- Feature engineering (6 new predictive features)
- Machine Learning model development
  - Logistic Regression (AUC: 0.96+)
  - Random Forest Classifier (AUC: 0.97+)
- Comprehensive model comparison and evaluation
- Feature importance analysis
- Customer risk segmentation (K-Means clustering)
- Individual customer risk scoring (0-100 scale)
- Business recommendations

**Key Achievements:**
- **97.25% AUC score** in churn prediction
- Successfully identifies **77% of customers who will churn**
- Identified top 5 churn predictors with importance scores
- Segmented 10,127 customers into 4 distinct risk categories
- Created actionable risk scores for every customer
- Quantified business impact: **$490K annual revenue protection**

---

## 📊 Dataset

**Source:** Kaggle - Credit Card Customers Dataset  
**Records:** 10,127 customers  
**Features:** 21 variables (18 original + 3 derived)  
**Target:** Customer attrition status (Churned / Retained)

### **Feature Categories:**

**Demographics:**
- Customer Age, Gender, Dependent Count
- Education Level, Marital Status, Income Category

**Account Information:**
- Card Category, Credit Limit
- Months on Book (customer tenure)

**Behavioral Metrics:**
- Total Relationship Count (number of products)
- Months Inactive (12 months period)
- Contact Count (12 months period)

**Transaction Data:**
- Total Transaction Amount & Count
- Transaction changes (Q4 vs Q1)
- Average Utilization Ratio

**Financial Metrics:**
- Total Revolving Balance
- Average Open to Buy

---

## 🔬 Methodology

### **Phase 1: Data Understanding (Notebook 1)**
- Initial data loading and inspection
- Basic statistical analysis
- Data structure validation
- Preliminary churn identification

### **Phase 2: Exploratory Analysis (Notebook 2)**
- Data quality assessment (0 missing values)
- Correlation matrix analysis
- Group comparisons (t-tests, statistical significance)
- Distribution analysis
- Comprehensive visualization suite
- Demographic and behavioral pattern discovery

### **Phase 3: Predictive Modeling (Notebook 3)**
1. **Feature Engineering**
   - Engagement Score (transactions per month)
   - Inactivity Rate (% of year inactive)
   - High Utilization flag (>70% credit usage)
   - Low Transaction flag (<50 transactions)
   - Few Products flag (≤2 products)
   - Declining Spending indicator

2. **Model Development**
   - 80-20 train and test stratified split
   - Feature standardization (StandardScaler)
   - Categorical encoding (LabelEncoder)
   - Cross validation for robustness

3. **Classification Models**
   - **Logistic Regression:** Interpretable baseline
   - **Random Forest:** Advanced ensemble method

4. **Model Evaluation**
   - ROC AUC curves
   - Confusion matrices
   - Precision, Recall, F1 scores
   - Feature importance ranking

5. **Customer Segmentation**
   - K Means clustering (4 segments)
   - Risk stratification (Low, Medium, High, Critical)
   - Segment profiling

---

## 🎯 Key Findings

### **1. Churn Rate**
- **16.1%** of customers have churned (1,627 out of 10,127)
- Significant business impact requiring immediate attention

### **2. Top 5 Churn Predictors** (by importance)
1. **Total_Trans_Ct** (40.2% importance) - Transaction count is the strongest predictor
2. **Total_Trans_Amt** (12.5% importance) - Transaction amount matters
3. **Total_Ct_Chng_Q4_Q1** (9.8% importance) - Declining transaction patterns
4. **Total_Revolving_Bal** (7.3% importance) - Credit usage patterns
5. **Contacts_Count_12_mon** (5.1% importance) - Customer service interactions

### **3. Behavioral Differences**

| Metric | Churned Customers | Retained Customers | Difference |
|--------|-------------------|-------------------|------------|
| **Avg Transactions** | 44.7 | 68.3 | -23.6 (34% less) |
| **Avg Transaction Amount** | $3,095 | $4,404 | -$1,309 (30% less) |
| **Months Inactive** | 2.3 | 2.1 | +0.2 (10% more) |
| **Products Owned** | 3.4 | 3.8 | -0.4 (11% fewer) |
| **Contact Count** | 2.5 | 2.3 | +0.2 (9% more) |

### **4. Model Performance**

| Model | Accuracy | AUC Score | Precision | Recall |
|-------|----------|-----------|-----------|--------|
| Logistic Regression | 91.2% | 0.9643 | 0.89 | 0.73 |
| **Random Forest** | **92.4%** | **0.9725** | **0.91** | **0.77** |

**Winner:** Random Forest (Superior performance across all metrics)

### **5. Customer Segmentation**

| Segment | Size | Churn Rate | Risk Level | Characteristics |
|---------|------|------------|------------|-----------------|
| 0 | 2,847 (28%) | 8.2% | 🟢 Low | High transactions, low inactivity |
| 1 | 3,214 (32%) | 12.5% | 🟡 Medium | Moderate activity, average engagement |
| 2 | 2,456 (24%) | 19.8% | 🟠 High | Declining transactions, rising inactivity |
| 3 | 1,610 (16%) | 31.2% | 🔴 Critical | Low transactions, high inactivity |

**Critical Insight:** Segment 3 (1,610 customers) requires immediate retention intervention

### **6. Risk Scoring Distribution**
- **Critical Risk (75-100):** 1,245 customers
- **High Risk (50-75):** 2,018 customers
- **Medium Risk (25-50):** 3,891 customers
- **Low Risk (0-25):** 2,973 customers

**Total Priority Customers:** 3,263 (Critical + High risk)

---

## 💼 Business Recommendations

### **1. Immediate Actions (Next 30 Days)**
- **Target:** 3,263 critical and high risk customers
- **Actions:**
  - Personalized retention outreach
  - Exclusive fee waivers or cashback bonuses
  - Dedicated relationship manager assignment for top 500 at risk accounts
  
**Expected Impact:** Save 25-30% of atr isk customers (~815-980 customers), protecting **$407K-$490K annual revenue**

### **2. Predictive System Deployment**
- Deploy Random Forest model for monthly customer scoring
- Integrate risk scores into CRM system
- Automated weekly reports for retention team
- Real-time alerts for customers crossing risk thresholds

### **3. Early Warning System**
**Implement automated triggers:**
- Alert when customer becomes inactive for 60+ days
- Flag when transaction count drops >30% quarter over quarter
- Monitor customers below 40 annual transactions
- Track product relationship reductions

### **4. Engagement & Product Strategy**
**Cross Selling Initiatives:**
- Current average: 3.5 products per customer
- Target: 5+ products per customer
- Focus: Customers with ≤2 products (highest churn risk)

**Transaction Incentives:**
- Rewards program based on transaction frequency
- Cashback tiers for monthly transaction milestones
- Merchant partnerships for exclusive cardholder discounts

### **5. Continuous Improvement**
- Quarterly model retraining with new data
- A/B test retention campaigns
- Track intervention effectiveness
- Measure ROI (typical retention programs achieve 300-500% ROI)
  
---

## 🛠️ Technologies Used

- **Python 3.x**
- **Jupyter Notebook**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **scikit learn** - Machine learning algorithms
  - LogisticRegression, RandomForestClassifier
  - KMeans clustering
  - StandardScaler, LabelEncoder
  - train_test_split, cross validation
  - ROC AUC, confusion matrix, classification metrics
- **scipy** - Statistical analysis
---

## 📂 Repository Structure

```
Credit-Card-Customers-Analysis/
│
├── Basic data set analysis.ipynb
│   └── Initial data exploration and understanding
│
├── Exploratory Data Analysis.ipynb
│   └── In depth EDA, correlations, visualizations
│
├── Predictive Modeling & Advanced Analysis.ipynb
│   └── ML models, segmentation, risk scoring
│
├── BankChurners.csv
│   └── Dataset (10,127 customers × 21 features)
│
├── README.md
│   └── Project documentation (this file)
│
└── requirements.txt
    └── Python package dependencies
```

## 🚀 How to Run

### **Prerequisites**

Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit learn scipy jupyter
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### **Running the Analysis**

1. **Clone or Download this Repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-churn-analysis.git
   cd credit-card-churn-analysis
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Run the Notebooks in Order**
   - **First:** `Basic data set analysis.ipynb` - Get familiar with the data
   - **Second:** `Exploratory Data Analysis.ipynb` - Deep dive into patterns
   - **Third:** `Predictive Modeling & Advanced Analysis.ipynb` - Build models

4. **Execute Cells**
   - Click "Cell" → "Run All" in each notebook
   - Or press `Shift + Enter` to run cells individually
     
---

## 📈 Results Summary

### **Analysed Question Answered:** ✅ YES!

**"What factors cause credit card customers to churn and can we identify at risk customers before they leave?"**

**Answer:**

1. **Top Churn Factors Identified:**
   - Low transaction count (40% of prediction power)
   - Decreasing transaction amounts (12.5%)
   - High inactivity periods (negative correlation)
   - Fewer banking products (11% fewer)
   - Increasing customer service contacts

2. **Prediction Capability:**
   - **97.25% AUC accuracy** in identifying churn risk
   - Successfully identifies **77% of customers who will churn**
   - Enables proactive intervention before customers leave

3. **Actionable Segmentation:**
   - 4 distinct risk segments created
   - **3,263 customers** flagged for immediate attention
   - Individual risk scores (0-100) for all customers

4. **Business Impact:**
   - **$490K annual revenue** can be protected
   - 15-20% projected churn reduction
   - 300-500% ROI on retention programs

**Answer Summary:**
✅ **97.25% AUC accuracy** in churn prediction  
✅ Successfully identifies **77% of customers who will churn**  
✅ **3,263 customers** flagged for immediate retention action  
✅ **$490,000/year** potential revenue protection

---
## 📊 Visualizations

The analysis includes many professional visualizations across all notebooks:
- Churn distribution charts (pie and bar)
- Correlation heatmaps
- Box plots comparing churners vs. retained customers
- Demographic breakdowns by churn status
- Transaction pattern analysis
- ROC curves for model performance
- Feature importance charts
- Customer segment distributions
- Risk score distributions
- Confusion matrices

*All visualizations are embedded in the Jupyter notebooks*

---

## 👤 Author

**Venuri Abeysekara**  
February, 2026

---

## 🙏 Acknowledgments

- **Dataset Source:** Kaggle - Credit Card Customers Dataset (Sakshi Goyal)
- **Methodology:** Industry best practices in churn prediction and customer analytics

