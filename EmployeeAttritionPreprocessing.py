#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import accuracy_score
import seaborn as sns

class preprocessing :
  
    # Constant variables
    JOB_ROLE_DICT = {
        'Human Resources': 8,
        'Sales Executive': 7,
        'Research Scientist': 6,
        'Laboratory Technician': 5,
        'Manufacturing Director': 4,
        'Healthcare Representative': 3,
        'Manager': 2,
        'Sales Representative': 1,
        'Research Director': 0,
        'other': 0
    }

    EDUCATION_FIELD_DICT = {
        'Life Sciences': 1,
        'Medical': 2,
        'Marketing': 3,
        'Technical Degree': 4,
        'Other': 5,
        'Human Resources': 6,
        'other': 0
    }

    def categorize_job_role(x: str) -> int:
        """Categorize job role"""
        return preprocessing.JOB_ROLE_DICT.get(str(x), preprocessing.JOB_ROLE_DICT['other'])

    def categorize_education_field(x: str) -> int:
        """Categorize education field"""
        return preprocessing.EDUCATION_FIELD_DICT.get(str(x), preprocessing.EDUCATION_FIELD_DICT['other'])

    # Load data
    
    def correlation_analysis(input_data):
        # Calculate the correlation matrix
        corr_matrix = input_data.corr()

        # Plot the heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')

        # Save the plot as a PNG file
        plt.savefig('correlation_heatmap.png')
    
    def processing_data(file_name, output_file, plot_correlation_analysis) : 
        attrdata = pd.read_csv(file_name)

        # Apply categorization functions
        attrdata['New JobRole'] = attrdata["JobRole"].apply(preprocessing.categorize_job_role)
        attrdata['New EducationField'] = attrdata["EducationField"].apply(preprocessing.categorize_education_field)

        # One-hot encoding for categorical variables
        business_travel = pd.get_dummies(attrdata["BusinessTravel"], dtype=int)
        job_role = pd.get_dummies(attrdata["JobRole"], dtype=int)
        marital_status = pd.get_dummies(attrdata["MaritalStatus"], dtype=int)

        # Concatenate one-hot encoded variables
        attrdata = pd.concat([attrdata, business_travel, job_role, marital_status], axis=1)

        # Map categorical variables to numerical variables
        attrdata["Gender_Num"] = attrdata["Gender"].map({"Female": 1, "Male": 0})
        attrdata["Over18_Num"] = attrdata["Over18"].map({"Y": 1, "": 0})
        attrdata["OverTime_Num"] = attrdata["OverTime"].map({"Yes": 1, "No": 0})
        attrdata["Attrition_numeric"] = attrdata["Attrition"].map({"Yes": 1, "No": 0})

        # Save data to csv
        attrdata.to_csv(output_file , index=False)
        
        if plot_correlation_analysis:
            
            preprocessing.correlation_analysis(attrdata)
            
if __name__ == "__main__":
    
    input_file = "HR_Analytics.csv" #define csv file for the input
    output_file = "Employee.csv" #define output file name
    plot_correlation_analysis = False # indicator to perform correlation analysis
    preprocessing.processing_data(input_file,output_file, plot_correlation_analysis)
plt.show()


# In[ ]:




