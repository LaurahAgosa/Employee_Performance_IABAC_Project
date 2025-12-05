
# Employee Performance Analysis for INX Future Inc.



Candidate Name : LIBESE LAURAH AGOSA

Candidate E-Mail : agosalaurah@gmail.com

Project Code : 10281

Assesment ID : E10901-PR2-V18

Module : Certified Data Scientist - Project

Exam Format : Open Project- IABAC™ Project Submission

Project Assessment :IABAC™ 

Registered Trainer : Ashok Kumar A

Submission Deadline Date: 05-Dec-2025

## Project Summary

Business Goal of the Project:
The objective of this project was to train and develop a machine learning model capable of predicting employee performance based on a set of demographic, experience-based, and job-related features.

This data science project analyzed employee performance to achieve the following goals:

1. Examine performance across different departments.

2. Identify the top three most important factors affecting employee performance.

3. Build a trained model that can predict employee performance based on selected input factors for use in hiring and HR decision-making.

4. Provide recommendations to improve employee performance based on insights from the analysis.

The given dataset contained 1,200 records and 28 features (1200 × 28), consisting of 19 numerical and 8 categorical variables. EmpNumber, an alphanumeric identifier, was excluded from modeling as it did not contribute to predicting performance. The target variable was ordinal in nature, supporting the classification framing of the project.

The analysis involved extensive Exploratory Data Analysis (EDA), including univariate, bivariate, and multivariate techniques to understand how different features relate to each other and to the performance rating. These steps helped answer the project goals regarding department-level performance, the key drivers of performance, and the most relevant factors for prediction.

From the correlation analysis, EmpEnvironmentSatisfaction and EmpLastSalaryHikePercent showed the strongest positive relationships with performance. WorkLifeBalance had a minor positive effect, while Age, TotalWorkExperience, and YearsAtCompany showed weak negative correlations. However, experience-related variables were strongly intercorrelated with each other, indicating natural career progression patterns.

Multiple machine learning models were developed and compared, including Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest Classifier. Based on performance metrics,particularly ROC-AUC,the Random Forest Classifier emerged as the best model due to its superior predictive capability.

One key aim of the project was to identify the most important features influencing performance ratings. Using the Random Forest model with Gini importance, the most influential factors were:

EmpEnvironmentSatisfaction (23.6%)

EmpLastSalaryHikePercent (22.7%)

YearsSinceLastPromotion

ExperienceYearsInCurrentRole

EmpJobRole

These features captured employees’ workplace satisfaction, salary growth, experience, and job function, all strongly tied to performance outcomes.

On the other hand, variables such as Gender, Attrition, and BusinessTravelFrequency had very low importance scores, indicating minimal influence on predicting performance.

Data preprocessing involved several important techniques, including label encoding, manual encoding, and frequency encoding, which were necessary for converting categorical attributes into numeric formats. Since machine learning algorithms generally require numerical data, these transformations were crucial for model training.

Overall, the project successfully achieved its goals using a combination of exploratory analysis, feature engineering, visualization techniques, and machine learning models.

1. REQUIREMENT

The employee dataset for this project was from IABAC and it was sourced from the IABAC™. The employee dataset is based on INX Future Inc referred as INX , which is one of the leading data analytics and automation solutions provider with over 15 years of global business presence. INX is consistently rated as top 20 best employers past 5 year. However, the dataset does not represent real employee records; it is a learning dataset designed for analytical and machine learning training purposes.

All analyses, visualizations, and model development were performed using Python within VS Code, utilizing the Jupyter Notebook extension. These tools provided an efficient environment for data preprocessing, model building, and performance evaluation throughout the project.

2. ANALYSIS

This involved describing the features present in the dataset, which paly a key role in anlysis. Features help to understand the relationship between independent and dependent variables. In order to understand the structure of the dataset and statistical summary, pandas was used. This dataset was divided into numerical and categorical features:

Numerical Features
    Age
    DistanceFromHome
    EmpHourlyRate
    NumCompaniesWorked
    EmpLastSalaryHikePercent
    TotalWorkExperienceInYears
    TrainingTimesLastYear
    ExperienceYearsAtThisCompany
    ExperienceYearsInCurrentRole
    YearsSinceLastPromotion
    YearsWithCurrManager

Ordinal Features
    EmpEducationLevel
    EmpEnvironmentSatisfaction
    EmpJobInvolvement
    EmpJobLevel
    EmpJobSatisfaction
    EmpRelationshipSatisfaction
    EmpWorkLifeBalance
    PerformanceRating

Categorical Features
    EmpNumber
    Gender
    EducationBackground
    MaritalStatus
    EmpDepartment
    EmpJobRole
    BusinessTravelFrequency
    OverTime
    Attrition

3. EXPLORATORY DATA ANALYSIS

This involeved univariate, bivariate and multivariate analysis. 

Libraries used: Matplotlib and Seaborn
plots used: Violinplot, Countplots, histplot,boxplot, Barplot
Tip: All interpretations and key insights are written below the plots

Univariate Analysis: This helped us undertsand the distribution of the categorical features and also obtain their unique labels.

Bivariate Analysis: This helped us undertstand the categoricala nd numerical feature relationship with the target variable(Performance Rating).

Mutlivariate Analysis: This hepled us undertsand how two variable relate  with respect to the target variable.

Conclusion:
- Some variables were right-skewed such as total work experience in years indicatiing that most employees fall on the lower end of these values with a few higher extreme values. Others were left-skewed including EmpEducationLevel meaning the majority of employees score on the higher end of these satisfaction or rating metrics while others were symmetrical such as EmplyHourly rate suggesting most of that their values are more evenly spread around the center.

- From correlation, some features were positively correlatd with performance rating[EmpEnvironmentSatisfaction and EmpLastSalaryHikePercent].

4. DATA PREPROCESSING

This involved several steps to clean the data before training the models.

    Checking for missing values:  There were nos missing values detected in the dataset.

    Checking for duplicates : There were no duplicate values detected in the dataset.

    Droping unique features: Employee Number was dropped as it was an identifier column.

    Categorical data converions: This involved converting the categoical columns to numerical. Three major techniques were incorporated including label encoding for (Overtime and Attrition), mapping encoding for features with minimal labels and frequency based encoding through mapping for features with high number of labels.

    Outlier handling: This involved IQR(Interquartile range) technique to impute outliers for the features thta were skewed to achieve a normal distribution.

    
    Balancing the target variable: This involeved SMOTENC technique(Synthetic Minority Oversampling Technique for Numerical and Continuous features). It helped geenrate synthetic minority samples for our mixed data type in a realistic way without distorting the categorical features.


    Splitting data: This involved splitting the data into 80% for training and 20% for testing.

    Feature Transformation( Data Standardization): This involved scaling of the training variables with the help of standardscaler to achieve a standard deviation of 1 and mean of 0, enabling the data to assume normal distribution.


5. MACHINE LEARNING MODEL TRAINING, EVALUATION AND PREDICTION

Before training the models, initially the dependent and independent features weere deifined in the preprocessing steps. The dpendent feature(Y) is our target variable while the independt features(X) are our variables that inluence performance rating.

MODEL TRAINING

This process involved experimenting with three different algorithms. This were:

i. Support Vector Machine(SVM) model:

This is a supervised machine learning model/algorithm that tries to find the best boundary(hyperplane) that separates data into different classes. This involved experimenting with both the linear and rbf(radial basis function)kernels. 

ii. K Nearest Neighbors (KNN) Model: It is a supervised lazy learning model that wokrs by looking at the k closest data points (neighbors) to a new observation and predicting the target based on those neighbors.  By lazy learning – it doesn’t train a model upfront; it uses the whole training data at prediction time. This involved building the calssification model with different k(n_neighbors) from 5-9.

iii. Random Forest: This is an ensemble learning method that constructs multiple decision trees and combines their predictions to improve overall performance and reduce overfitting. It works by training each tree on a random subset of the data and using a random subset of features for each split, resulting in a diverse set of trees that generalize a better new data.

MODEL EVALUATION

This step involved assessing the performance of the traned models with different metrics including accuracy score, classification report, confusion matrix and ROC-AUC(Receiever Operating characteristsics-Area under curve).

i. Accuracy score: It involved determining the both the training and the testing accuracy. These were then compared to see if the model was overfitting. In all the models, the training scores identified the models to be performing well with mild overfitting noticed with the test score and threfore, hyperparameter tunning was performed. 
    
    a. Support Vector Machine: rbf kernel was selected as itachieved a higher training accuracy of 91% compared to that of 82% for the linear kernel. This suggested that the underlying relationships in the dataset are non-linear. However the training data for the rbf kernel had an accuracy score of 96% suggesting that the training data performs well and still laggs in the test data. After hypeparameter tunnig, although the test score dropped to 89%, it delivered a higher and more stable F1-score, indicating improved performance on the minority class.

    b. Random Forest Classifier: The training accuracy score was 1005 while that of test score was 95%. After hypeparameter tunning the score lowered to 83%. Although it reduced overfitting the original model was selected due to its high accuracy.

    c. K Nearest Neighbors: This involeved the use of different k from 5-9, and as k increased the accuracy dropped to 78% and remained stable. The model achieved the highest accuracy of 80% while k=5, suggesting that smaller values of k capture the data more effectively. However this was lower to the training accuracy of 85%. After hypeparameter tunning, the model performed better with an accuracy score of 84.76%. This was achieved when optimal parameters were n_neighbors=1, metric=manhattan and weight=uniform.

ii. Confusion matrix: This involved identifying how the model correctly places employees in different performance levels with minimal miscalculations.  Random Forest had the most correct classification of employees across all the three ratig categories. It showed strong performance for ratings 2, with minimal confusion across neighbouring classes. This was then followed by SVM rbf that showed strong performance for ratings 3, with minimal confusion across neigbouring classes. KNN had the highets level of confusion especially for ratings 3 where the model frequently misclassified the employees as either ratings 2 or 4.

iii. ROC-AUC(Receiever Operating characteristsics-Area under curve): This involved using one-vs-rest and one-vs-one performance categories. Random forest calssifier achieved the highest ROC-AUC-OvR and ROC-AUC-OvO(99.49%) which wa also consistent. This indicated it reliably identifies employees performance based on the different levels of rating. KNN had nearly identical ROC-AUC-OvR and ROC-AUC-OvO (0.8851 vs 0.8848) respectively. SVM rbf had high and consistent score between OvR and OvO (0.9818 vs 0.9822) respectively.

MODEL PREDICTION:
This involved creating a predictive sytem to determine how the model works when data is fed into the system. 

FEATURE IMPORTANCE:

This involved identifying the important features that drive employee performance. It was done after slecting the best performing model as Random Forest Classifier. It involved computing feature importance using built in Gini importance scores. This allows  Random Forest to naturally rank features based on how much they reduce impurity across all decision trees.

Using the Random Forest Classifier’s Gini importance method, the most influential features in predicting employee performance were EmpEnvironmentSatisfaction and EmpLastSalaryHikePercent, contributing 23.6% and 22.7% of the total importance respectively.


6. Saving the model
The random Forest classifier model was saved with the help of pickle file


Tools and Libraries used:

Tools:
Jupyter

Library Used:
Pandas
Numpy
Matplotlib
Seaborn
Sklearn
Pickle


GOALS ANALYSIS

GOAL 1: DEPARTMENT WISE PERFORMANCE


Plots used:

1. Violinplot- This shows how quantitative values are distributed within various categorical groups for easy comparison.

2. Countplots- This visualize the frequency counts of categorical observations using bar representations.

Insights from Department wise performance

Sales Department:
The countplot is dominant at Rating 3, strong Rating 4 presence, minimal low performers. Violin plot shows wide performance distribution with healthy density across both genders. Therefore, sales has high-performing team with diverse skill levels, shows both consistency (peak at 3) and capability for excellence with good representation at 4.

Human Resources:
The count plot has strong concentration at Rating 3 and moderate Rating 4. Violin plots has a tight, concentrated distribution around median performance for both genders. This suggests consistent, reliable performers, team delivers stable output with limited performance variation.

Development:
This has a heavy concentration at Rating 3 and  good Rating 4 representation as per the countplot. It also has a balanced distribution with good spread across performance levels for both genders. This indicates strong technical team with quality output, maintains high standards while accommodating some performance diversity

Data Science:
This demonstrates the lowest overall performance among all departments, with greater representation in lower rating tiers and less concentration in top performance categories compared to other departments.

Research & Development:
This has strong peak at Rating 3 and healthy Rating 4 presence seen. Violoin plots is concentrated around high-performance ratings with limited outliers. Therefore it is focused high-achievers,team maintains excellent standards with minimal performance gaps

Finance:
Has a clear dominance of Rating 3 and  good secondary Rating 4 presence seen. The violin plot shows consistent, predictable performance distribution. This suggests a stable, dependable team  that delivers reliable results with consistent quality.


GOAL 2: TOP 3 IMPORTANT FACTORS AFFECTING EMPLOYEE PERFORMANCE

The top three important features affecting the performance rating were:

Employment Environment Satisfaction
Employee Salary Hike Percentage
Employee Work Life Balance


From the correlation heatmap matrix and crosstabs, these were the findings:

1. Employee Environment Satisfaction

- Employee environment satisfaction appeared to be strongly associated with performance. Low-satisfaction employees (levels 1 and 2) contribute the highest number of low performers, while highly satisfied employees (levels 3 and 4) contribute most of the high performers. This suggested that a supportive work environment positively influences performance.

2. Employee Last Salary Hike Percent

- Salary hikes of 20–22% had the highest concentration of top performers, while smaller increments (11–19%) were dominated by average or below-average performers. This suggested that performance-based salary increments may be effective and that higher hikes are typically given to high-performing employees.

3. Employee Work Life Balance

- Employees with the poorest work–life balance (level 1) had no high performers, while levels 3 and 4 include the majority of the top performers. This indicated that maintaining a healthy work–life balance supports higher employee performance.


GOAL 3: A TRAINED MODEL WHICH CAN PREDICT EMPLOYEE PERFORMANCE

This project used machine learning algorithm to come up with the predictive model. Several models were experimented and after performance metrics ealuation, Random Forest Classifier was selected as the final model for this task.


GOAL 4: RECOMMENDATIONS TO IMPROVE EMPLOYEE PERFORMANCE

In order to achieve the improbement of employee performance, the company needs to focus on:

1. Improving work place environment, more focus to be on the employee environment satisfaction

2. Reviewing salary hike polcies, this will give a boost to the employees to perform well

3. Shorten promotion cycles and create clear career pathways. This can be achieved by reducing year since last promotion to months.

4. Incorporate role specific traininga nd mentoring. This ensures employee's effectiveness in their current role is adequatetly achieved.

5. Ensure overtime is monitored and promotion of work-life balance. This help reduce excessive overtime that is harming sustained performance.

6. The company should put more focus on development and sales department as they have an overall high performance compared to the rest of the departments.














