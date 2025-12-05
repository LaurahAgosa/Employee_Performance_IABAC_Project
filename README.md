📊 Employee Performance Analysis & Prediction – INX Future Inc.

Author: Libese Laurah Agosa
Email: agosalaurah@gmail.com

Certification Project: IABAC – Certified Data Scientist
Assessment ID: E10901-PR2-V18
Dataset Source: IABAC™ Learning Dataset
Tools Used: Python, Jupyter Notebook, Scikit-Learn, Pandas, Matplotlib, Seaborn

📌 Project Overview

This project focuses on analyzing and predicting employee performance using demographic, experiential, and job-related factors. It was completed as part of the IABAC Certified Data Scientist assessment.

Using a dataset of 1,200 employees with 28 features, I explored key performance drivers, conducted department-level analysis, and built predictive machine learning models to classify employee performance ratings.

🎯 Objectives
General Objective

Develop a predictive machine learning model to estimate employee performance based on structured HR data.

Specific Objectives

Clean, explore, and visualize the dataset.

Identify key factors influencing employee performance.

Analyze department-wise performance distribution.

Train and evaluate multiple machine learning models.

Recommend actionable strategies to improve performance.

📁 Dataset Description

1200 rows × 28 features

Target variable: PerformanceRating (ordinal)

Feature types:

Numerical: Age, ExperienceYearsInCurrentRole, EmpHourlyRate, etc.

Ordinal: JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance

Categorical: Department, JobRole, Gender, BusinessTravelFrequency

Excluded variable: EmpNumber (identifier, no predictive value)

🔍 Exploratory Data Analysis (EDA)

The analysis included univariate, bivariate, and multivariate visualizations using:

Barplots / Countplots

Violin plots

Boxplots

Histograms

Correlation matrices

Key Insights

Environment satisfaction and salary hike percentages are strong positive indicators of performance.

Experience-related variables are highly intercorrelated (natural career progression).

Most departments peak at Rating 3, but Sales, Development, and R&D show strong representation of Rating 4 performers.

🛠 Data Preprocessing Steps

Handled categorical encoding (Label Encoding, Mapping, Frequency Encoding)

Removed irrelevant unique features

Outlier handling using IQR method

Target balancing using SMOTENC for mixed data

Feature scaling using StandardScaler

80/20 Train-Test split

🤖 Machine Learning Models

Three models were trained and compared:

Model	Best Accuracy	ROC-AUC OvR	Notes
Random Forest Classifier	95%	0.9949	Best performing model
Support Vector Machine (rbf)	89%	0.9818	Good non-linear performance
K-Nearest Neighbors (KNN)	84.76%	0.8851	Sensitive to k selection
✔ Best Model: Random Forest Classifier

Random Forest achieved the highest accuracy and ROC-AUC, demonstrating robust classification across all rating categories.

⭐ Feature Importance (Top Drivers of Performance)

According to Random Forest Gini Importance:

EmpEnvironmentSatisfaction – 23.6%

EmpLastSalaryHikePercent – 22.7%

YearsSinceLastPromotion

ExperienceYearsInCurrentRole

EmpJobRole

These features demonstrate that workplace satisfaction, salary growth, and career progression significantly affect performance outcomes.

🏢 Department-wise Performance Analysis
🔹 Sales

High proportion of Rating 3 and 4 performers; strong performance culture.

🔹 Development

Consistent high performance with good rating distribution.

🔹 Research & Development

Clustered around high performance; very few low performers.

🔹 Human Resources

Stable and consistent performers, mostly at Rating 3.

🔹 Data Science

Lowest performance distribution—needs targeted support.

📝 Recommendations

To improve performance at INX Future Inc., the company should:

Enhance workplace environment (top predictor of performance).

Review salary hike policies to reward and motivate high achievers.

Shorten promotion cycles and clarify career growth pathways.

Offer role-specific training and mentorship.

Promote work-life balance and monitor overtime.

Focus development strategies on low-performing departments such as Data Science.

💾 Model Deployment

The Random Forest model was saved using pickle for future predictions and integration into applications such as:

Employee evaluation tools

HR dashboards

Streamlit-based ML applications

📚 Technologies & Libraries

Python

Jupyter Notebook

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn

SMOTENC (imbalanced-learn)

Pickle

📂 Repository Structure (Suggested)
├── README.md
├── employee_performance.ipynb
├── performance_summary.md
├── data/
│   └── employee_data.csv
├── models/
│   └── random_forest_model.pkl
├── images/
│   └── plots_from_EDA/

📌 Conclusion

This project successfully developed a data-driven framework for analyzing and predicting employee performance.
The insights gained can support HR teams in:

Workforce planning

Employee development

Promotions and salary review

Organizational productivity strategies

The Random Forest model delivered high predictive accuracy and identified clear performance drivers, making it suitable for real-world decision support systems.
