# Tata-Steel-Machine-Failure-Prediction

## Project Overview

This project aims to develop a machine learning model that predicts machine failures in advance to enable proactive maintenance, minimize downtime, and improve operational efficiency. The primary focus is to build, tune, and evaluate multiple machine learning models, with an emphasis on optimizing **Recall** and **ROC-AUC** scores to detect machine failures as early as possible.

## Key Highlights
- **Best Model:** Random Forest Classifier
- **Performance:** 
  - **Recall:** 84.04%
  - **ROC AUC:** 95.54%

---

## Table of Contents
- [Dataset Variables](#dataset-variables)
- [Built With](#built-with)
- [Steps Involved For Analysis](#steps-involved-for-analysis)
- [Key Findings and Implications](#key-findings-and-implications)

---


### Dataset Variables:

#### Machine Operational Variables:

- **Type**: Indicates the quality of the product, classified into categories such as Low, Medium, or High.
- **Air Temperature [K]**: Represents the air temperature, simulated using a random process with variability around a standard value.
- **Process Temperature [K]**: Represents the temperature within the process, slightly higher than the air temperature, with its own variability.
- **Rotational Speed [rpm]**: Describes the speed at which the machine operates, calculated based on a fixed power level with added random variation.
- **Torque [Nm]**: Measures the force applied by the machine, distributed around an average value with specific variation.
- **Tool Wear [min]**: Indicates the wear on the tool, which increases based on the product quality category.
- **Machine Failure**: Indicates whether the machine has experienced a failure. Several types of failures are described.
  
- **Tool Wear Failure**: Occurs when the tool is replaced or fails after a certain amount of usage time, which is randomly determined within a specific range.
  
- **Heat Dissipation Failure**: Happens if the temperature difference between the air and process is too small, and the machine speed is below a certain threshold.
  
- **Power Failure**: Occurs when the power required for the process (calculated from torque and speed) falls outside of an acceptable range.
  
- **Overstrain Failure**: Happens when the combined effect of tool wear and torque exceeds specific limits, based on the product quality.
  
- **Random Failures**: Represents a small probability of failure occurring randomly, independent of other process parameters.

---


## Built With

This project was developed using the following technologies:

- **Python**
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization and generating charts.
- **Seaborn**: For advanced statistical data visualization.
- **Scikit-learn**: For building, training, and evaluating machine learning models.
- **Google Colab**: Cloud-based platform used to run Jupyter notebooks.

---


## Steps Involved For Analysis

1. **Dataset Inspection**  
   - Initially, the dataset was examined to identify its structure, data types, and any issues (e.g., missing values, duplicates).

2. **Data Preprocessing**  
   - **Missing Values**: Handled missing data by filling in or removing rows with missing values.
   - **Feature Scaling**: Applied standardization or normalization to numerical features to improve model performance.
   - **Categorical Encoding**: Categorical data (like machine type or failure type) was encoded into numerical values.

3. **Exploratory Data Analysis (EDA)**  
   - **Univariate Analysis**: Analyzed the distribution of individual features (e.g., using histograms and boxplots for numerical features, and bar charts for categorical features).
   - **Bivariate Analysis**: Explored relationships between pairs of variables (e.g., scatter plots for numerical variables, heatmaps for correlation).
   - **Correlation Analysis**: Investigated correlations between numerical features using correlation matrices to understand which features are strongly related.
   - **Feature Distributions**: Visualized the distribution of operational variables (e.g., Air Temperature, Torque) to check for skewness, outliers, and patterns.
   - **Failure Type Analysis**: Visualized and compared the occurrence of different failure types based on operational variables.

4. **Model Selection**  
   - **Logistic Regression**: Used as a baseline model.
   - **Decision Tree**: Selected to handle non-linear relationships and for interpretability.
   - **Random Forest**: Chosen for its robustness, ability to handle large datasets and superior performance.

5. **Hyperparameter Tuning**  
   - Optimized model hyperparameters using **GridSearchCV** to achieve the best performance.

6. **Model Evaluation**  
   - Focused on **Recall** and **ROC AUC** as evaluation metrics. Recall was emphasized to minimize false negatives (critical for detecting machine failures).

7. **Feature Importance Analysis**  
   - **Random Forest** was used to identify key features influencing machine failures. This helped in understanding which operational factors are most predictive of failures.

8. **Model Deployment**  
   - The best-performing model (Random Forest with tuned hyperparameters) was saved and exported as a **Pickle** file (`best_model_rf.pkl`) for deployment.


## Key Findings and Implications

### Model Performance

- **Best Model Selection**:  
  The **Random Forest model** emerged as the best performer, achieving:
  - **Recall**: 84.04%  
  - **ROC AUC**: 95.54%
 


- **Suggestions**:  
  Random Forest is the optimal model for this predictive maintenance task due to its ability to handle non-linear relationships, its robustness against overfitting, and its high performance on the key metrics.

### Importance of Recall and ROC AUC
- **Findings**:  
  - **Recall**: A higher recall score ensures that the model catches as many failures as possible, which is crucial in the context of predictive maintenance where failing to identify a failure could lead to costly downtime.
  - **ROC AUC**: A higher ROC AUC indicates that the model performs well at distinguishing between failure and non-failure events across different thresholds.

- **Suggestions**:  
  Prioritize models with high recall to reduce the risk of missing potential failures. ROC AUC is also important for assessing overall model performance across various decision thresholds.

### Feature Engineering Insights
- **Findings**:  
  The most important features influencing failure predictions were operational variables like **Power**, **Heat Dissipation Failure (HDF)**, and **Overstrain Failure (OSF)**. These variables were critical in predicting whether a machine was likely to fail.

- **Suggestions**:  
  Focus on monitoring and improving the operational parameters related to **Power**, **HDF**, and **OSF**. These factors should be prioritized in maintenance and monitoring schedules to prevent failures.

### Model Performance Post-Tuning
- **Findings**:  
  Hyperparameter tuning with GridSearchCV improved the performance of the Random Forest model significantly:
  - **Recall**: Increased by 5.4%
  - **ROC AUC**: Increased by 1.96%

- **Suggestions**:  
  Hyperparameter optimization is critical in improving the model’s predictive ability. Fine-tuning the model's parameters can lead to better results in real-world scenarios.


### Business Impact
- **Findings**:  
  The ability to accurately predict machine failures allows businesses to:
  - **Minimize Downtime**: Proactively schedule maintenance before machines fail.
  - **Reduce Operational Costs**: Avoid costly unscheduled repairs and downtime.
  - **Optimize Maintenance Schedules**: Use predictive insights to allocate resources efficiently and only when needed.

- **Suggestions**:  
  Deploying this model will result in improved operational efficiency, cost savings, and a more streamlined maintenance process. Businesses can utilize this tool to stay ahead of potential issues and avoid disruptions in their production schedules.

---




















