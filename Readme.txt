Final Project - German Bank Loan Default Prediction
****************************************************

Project Overview
****************
-> This project focuses on developing a Machine Learning model to predict whether a bank customer 
   will default on their loan based on historical data. 
-> The dataset utilized contains diverse information regarding credit applicants, encompassing attributes 
   like credit history, job status, purpose of the loan, and more.


Data Description: 
****************
-> The bank has historical information on relevant features for each customer such as employment duration, existing loans count, 
   saving balance, percentage of income, age, default status. 

-> The data set has 17 columns and 1000 rows. Columns are described below and each row is a customer. 

|- checking_balance - Amount of money available in account of customers
|- months_loan_duration - Duration since loan taken
|- credit_history - credit history of each customers
|- purpose - Purpose why loan has been taken
|- amount - Amount of loan taken
|- savings_balance - Balance in account
|- employment_duration - Duration of employment
|- percent_of_income - Percentage of monthly income
|- years_at_residence - Duration of current residence
|- age - Age of customer
|- other_credit - Any other credits taken
|- housing- Type of housing, rent or own
|- existing_loans_count - Existing count of loans
|- job - Job type
|- dependents - Any dependents on customer
|- phone - Having phone or not
|- default - Default status (Target column)
*********************************************************************

Contents
------------
 1. Dataset:
    The dataset used in this project is "credit.csv". It contains various attributes of credit applicants, including 
    both numerical and categorical features.

 2 .Preprocessing:
    The preprocessing steps involve handling missing values, encoding categorical variables, and scaling numerical
    features. Additionally, some feature engineering may have been performed to improve model performance.

 3. Exploratory Data Analysis (EDA):
    Exploratory data analysis was performed to understand how the data is spread out, the connections
    between features and any potentialtrends. Visual aids like histograms, box plots and count plots 
    were utilized to examine the dataset.

 4. Model Building (Without Hyperparameter  Tuning):
    These include Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, 
    Decision Tree Classifier, Gradient Boosting Classifier, XGB Classifier, and SGD Classifier.

 5. Model Evaluation (Without Hyperparameter  Tuning):
    To get insights about the models before Tuning

 6. Model Building (With Hyperparameter  Tuning) :
    Building Model using Hyperparameter tuning

 7. Model Evaluation (With Hyperparameter  Tuning):
    Model Evaluation using ROC and PR curve also Area under both curves calculated for all models also 
    confusion matrix for individual models also included.
                                                   
 8. Voting Classifier :
    By using Ensemble technique we have used voting classifier by combining selected better performing models.

 9. Results:
    The performance of the Voting Classifier was assessed using various metrics, and its classification report 
    and confusion matrix are provided for detailed analysis.

 10.Conclusion:
    The project concludes showing the effectiveness of ensemble techniques, particularly the Voting Classifier, 
    in predicting loan defaults by leveraging the combined effect of different machine learning algorithms.


   Getting Started with Project
   *****************************
   Step 1 - Unzip the folder named Final Project
   Step 2 - Make sure these files exist after unzipping - "Final Project.ipynb", credit.csv.
   Step 3 - Make sure below libraries are installed (The code given below is for Notebook type env):
            Pandas -        !pip install pandas
            NumPy  -       !pip install numpy
            Seaborn -      !pip install seaborn
            Matplotlib -   !pip install matplotlib
            Scikit-learn - !pip install scikit-learn
            XGBoost - !pip install xgboost
	    ****** You should run this code file in Jupyter or google colab like environment******
            ****** IMPORTANT  - If have them in your system make sure that they are updated to latest version ********
   Step 4 - Open Google colab or jupyter Notebook open the Final Project.ipynb before you run make sure the file path to the 
            credt.csv is correct while import using pandas
   Step 5 - In this step you can run the project using Runall option in the notebook(Steps for runall option may change depending on Notebook)
 