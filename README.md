# Credit-Card-Customers-Analysis
This project analyzes credit card customer churn patterns to identify at-risk customers and develop data-driven retention strategies.

## Business Problem
What factors cause credit card customers to churn, and can we identify at-risk customers before they leave?

## Dataset
- **Source:** Kaggle - Credit Card Customers Dataset
- **Records:** 10,127 customers
- **Features:** 21 variables (demographics, transactions, account info)

## Key Findings
- **Churn Rate:** 16.07% of customers have churned
- **Top Churn Predictor:** Months of inactivity
- **Key Insight:** Churned customers have:
  - 13 fewer transactions on average
  - 2.2 more months of inactivity per year
  - Fewer banking products
  - Lower spending ($651 less)

## Methodology
1. **Data Preprocessing** - Cleaned and prepared 10,127 customer records
2. **Exploratory Data Analysis** - Visualized churn patterns and customer behaviors
3. **Statistical Analysis** - Correlation analysis and group comparisons
4. **Visualization** - Created charts comparing churners vs existing customers

## Technologies Used
- **Python 3**
- **Jupyter Notebook**
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

## Project Structure
```
credit-card-churn-analysis/
│
├── Credit_Card_Customers_Analysis.ipynb    # Main analysis notebook
├── BankChurners.csv                         # Dataset
└── README.md                                # Project documentation
```

## Key Visualizations
- Churn distribution (pie chart and bar chart)
- Comparison of churners vs existing customers
- Feature correlation heatmap
- Behavioral metrics comparison

## Business Recommendations
1. **Monitor Inactivity:** Implement alerts for customers inactive >2 months
2. **Increase Engagement:** Launch re-activation campaigns for dormant accounts
3. **Promote Cross-Selling:** Encourage customers to use multiple banking products
4. **Transaction Incentives:** Reward frequent card usage to boost engagement

## How to Run
1. Install required libraries:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
2. Open Jupyter Notebook:
```bash
   jupyter notebook
```
3. Open `Credit_Card_Customers_Analysis.ipynb`
4. Run all cells (Cell → Run All)

## Results
✅ Successfully identified key churn drivers  
✅ Discovered significant behavioral differences  
✅ Provided actionable business recommendations  

## Author
**Venuri Abeysekara** 
Index No: 21020035


## Acknowledgments
- Dataset: Kaggle (Sakshi Goyal)


Your repository URL will look like:
```
https://github.com/YOUR_USERNAME/credit-card-churn-analysis
```

