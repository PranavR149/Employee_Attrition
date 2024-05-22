#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Display all rows in the DataFrame
pd.set_option('display.max_rows', None)

# Display all columns in the DataFrame
pd.set_option('display.max_columns', None)


# In[2]:

class AttritionData:
    """
    A class to handle attrition data.

    Attributes:
        file_path (str): The file path to the CSV containing attrition data.
        data (DataFrame): The DataFrame containing the loaded data.
    """
    def __init__(self, file_path):
        """
        Initialize the AttritionData object.

        Parameters:
            file_path (str): The file path to the CSV containing attrition data.
        """
        self.file_path = file_path
        self.data = None

    def read_data(self):
        """
        Read the data from the CSV file into a DataFrame.

        Returns:
            None
        """
        self.data = pd.read_csv(self.file_path)

# Initialize the `attrition` object and read the data
attrition = AttritionData("HR_Analytics.csv")
attrition.read_data()

# Check Null
print(attrition.data.info())

# Summary Statistics
print(attrition.data.describe())


# ##### Age Distribution

# In[3]:


class AgePlot:
    """
    A class to generate plots related to Demographic - Age Field analysis.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """    
    def __init__(self, data):
        """
        Initialize AgePlot object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data
        self.plot_save_dir = "EDA_PLOTS"  # Directory to save plots
        
        # Create the directory if it doesn't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def plot_age_distribution(self):
        """
        Plot the distribution of age.

        Returns:
            None
        """
        # Plotting the distribution of age
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data['Age'], kde=True)
        plt.title('Distribution of Age')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'age_distribution.png'))
        plt.close()
        
    def plot_age_vs_attrition_count(self):
        """
        Plot a countplot of Age with respect to Attrition.

        Returns:
            None
        """
        # Create subplots with a specific figure size
        fig, ax = plt.subplots(figsize=(15, 4))

        # Plot the countplot of Age with respect to Attrition
        sns.countplot(x='Age', hue='Attrition', data=self.data, palette='colorblind')
        plt.title('Agewise Attrition')
        
        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'age_vs_attrition_count.png'))
        
        plt.close()

# Function Call
age_plotter = AgePlot(attrition.data)
age_plotter.plot_age_distribution()
age_plotter.plot_age_vs_attrition_count()


# ##### Gender Analysis

# In[4]:


class GenderPlot:
    """
    A class to generate plots related to Gender analysis.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """
    def __init__(self, data):
        """
        Initialize GenderPlot object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data
        self.plot_save_dir = "EDA_PLOTS"  # Directory to save plots
        
        # Create the directory if it doesn't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def plot_gender_count(self):
        """
        Plot the distribution of age.

        Returns:
            None
        """
        # Plotting the count plot of Gender
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=self.data, x='Gender', palette='Set2')
        plt.title('Count Plot of Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')

        # Annotating each bar with its count value
        for p in ax.patches:
            count = p.get_height()  # Get the height of the bar (count value)
            # Annotate each bar with its count value, positioning the text slightly below the top of the bar
            ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., count * 0.95),  # Place text at 95% of the bar's height
                        ha='center', va='top',  # Center and top alignment
                        xytext=(0, -3),  # Offset the text by 3 points downward
                        textcoords='offset points')

        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'gender_count.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()
        
    def plot_attrition_by_gender(self):
        """
        Plot a 100% stacked bar plot of Attrition by Gender.

        Returns:
            None
        """
        # Group the data by Gender and Attrition and count the number of occurrences
        grouped_data = self.data.groupby(['Gender', 'Attrition']).size().reset_index(name='count')

        # Calculate the total count for each gender
        total_count = grouped_data.groupby('Gender')['count'].transform('sum')

        # Calculate the proportion of each category within each group
        grouped_data['proportion'] = grouped_data['count'] / total_count * 100

        # Plotting the 100% stacked bar plot of Gender with respect to Attrition
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=grouped_data, x='Gender', y='proportion', hue='Attrition', palette='Set2', estimator=sum)
        plt.title('100% Stacked Bar Plot of Gender with respect to Attrition')
        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)  # Set y-axis limit to 0-100 for percentage

        # Annotating each bar with its percentage value
        for p in ax.patches:
            percentage = p.get_height()  # Get the height of the bar (percentage value)
            # Annotate each bar with its percentage value, positioning the text in the middle of the bar
            ax.annotate(f'{int(percentage)}%', (p.get_x() + p.get_width() / 2., percentage / 2),  # Place text at half of the bar's height
                        ha='center', va='center',  # Center alignment
                        xytext=(0, 5),  # Offset the text by 5 points upward
                        textcoords='offset points')

        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'attrition_by_gender.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()


# Function Call
gender_plotter = GenderPlot(attrition.data)
gender_plotter.plot_gender_count()
gender_plotter.plot_attrition_by_gender()


# ##### Marital Status Analysis

# In[5]:


class MaritalStatusPlot:
    """
    A class to generate plots related to Marital Status analysis.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """
    def __init__(self, data):
        """
        Initialize MaritalStatusPlot object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data
        self.plot_save_dir = "EDA_PLOTS"  # Directory to save plots
        
        # Create the directory if it doesn't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def plot_marital_status_count(self):
        """
        Plot a count plot of Marital Status.

        Returns:
            None
        """
        # Plotting the count plot of MaritalStatus
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=self.data, x='MaritalStatus', palette='Set2')
        plt.title('Count Plot of Marital Status')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')

        # Annotating each bar with its count value
        for p in ax.patches:
            count = p.get_height()  # Get the height of the bar (count value)
            # Annotate each bar with its count value, positioning the text slightly above the top of the bar
            ax.annotate(f'{int(count)}', (p.get_x() + p.get_width() / 2., count),  # Position the text at the top of the bar
                        ha='center', va='bottom',  # Center and bottom alignment
                        xytext=(0, 2),  # Offset the text by 2 points upward
                        textcoords='offset points')

        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'marital_status_count.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def plot_marital_status_vs_attrition(self):
        """
        Plot a count plot of Marital Status with respect to Attrition.

        Returns:
            None
        """
        # Group the data by MaritalStatus and Attrition and count the number of occurrences
        grouped_data = self.data.groupby(['MaritalStatus', 'Attrition']).size().reset_index(name='count')

        # Plotting the count plot of MaritalStatus with respect to Attrition
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=grouped_data, x='MaritalStatus', y='count', hue='Attrition', palette='Set2')
        plt.title('Count Plot of Marital Status with respect to Attrition')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')

        # Annotating each bar with its count value
        for p in ax.patches:
            count = p.get_height()  # Get the height of the bar (count value)
            # Annotate each bar with its count value, positioning the text slightly above the top of the bar
            ax.annotate(f'{int(count)}', (p.get_x() + p.get_width() / 2., count),  # Place text at the top of the bar
                        ha='center', va='bottom',  # Center and bottom alignment
                        xytext=(0, 2),  # Offset the text by 2 points upward
                        textcoords='offset points')

        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'marital_status_vs_attrition.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def plot_100_percent_stacked_marital_vs_attrition(self):
        """
        Plot a 100% stacked bar plot of Marital Status with respect to Attrition.

        Returns:
            None
        """
        # Group the data by MaritalStatus and Attrition and count the number of occurrences
        grouped_data = self.data.groupby(['MaritalStatus', 'Attrition']).size().reset_index(name='count')

        # Calculate the total count for each MaritalStatus category
        total_count = grouped_data.groupby('MaritalStatus')['count'].transform('sum')

        # Calculate the proportion of each category within each MaritalStatus group
        grouped_data['proportion'] = grouped_data['count'] / total_count * 100

        # Pivot the data to create a wide-form DataFrame for 100% stacked bar plot
        pivoted_data = grouped_data.pivot(index='MaritalStatus', columns='Attrition', values='proportion').fillna(0)

        # Plotting the 100% stacked bar plot of Marital Status with respect to Attrition
        plt.figure(figsize=(8, 6))
        ax = pivoted_data.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], edgecolor='k')

        plt.title('100% Stacked Bar Plot of Marital Status with respect to Attrition')
        plt.xlabel('Marital Status')
        plt.ylabel('Percentage')

        # Annotating each bar with its percentage value
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.annotate(f'{int(height)}%', (x + width / 2, y + height / 2), ha='center', va='center')

        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'stacked_marital_vs_attrition.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()


# Function Call
marital_status_plotter = MaritalStatusPlot(attrition.data)
marital_status_plotter.plot_marital_status_count()
marital_status_plotter.plot_marital_status_vs_attrition()
marital_status_plotter.plot_100_percent_stacked_marital_vs_attrition()


# ##### Educational Field

# In[6]:


class EducationFieldPlot:
    """
    A class to generate plots related to Education Field analysis.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """
    def __init__(self, data):
        """
        Initialize EducationFieldPlot object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data
        self.plot_save_dir = "EDA_PLOTS"  # Directory to save plots
        
        # Create the directory if it doesn't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def plot_education_field_distribution(self):
        """
        Plot a pie chart of Education Field distribution.

        Returns:
            None
        """
        # Define labels for the pie chart
        labels = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']

        # Plotting the pie chart of Education Field distribution
        plt.figure(figsize=(8, 8))
        plt.pie(self.data['EducationField'].value_counts(), labels=labels, autopct='%.1f%%')
        plt.title('Education Field Distribution')
        
        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'education_field_distribution.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def plot_education_field_vs_attrition(self):
        """
        Plot pie charts for the distribution of EducationField with respect to Attrition.

        Returns:
            None
        """
        # Group the data by EducationField and Attrition, and calculate the count for each combination
        grouped_data = self.data.groupby(['EducationField', 'Attrition']).size().unstack()

        # Define labels for the pie charts
        labels = grouped_data.index

        # Calculate the percentage of each EducationField for Attrition=Yes and Attrition=No
        percentage_yes = grouped_data['Yes'] / grouped_data.sum(axis=1) * 100
        percentage_no = grouped_data['No'] / grouped_data.sum(axis=1) * 100

        # Plotting the pie chart for EducationField with respect to Attrition=Yes
        plt.figure(figsize=(8, 8))
        plt.pie(percentage_yes, labels=labels, autopct='%.1f%%', startangle=140, colors=sns.color_palette('Set2'))
        plt.title('Distribution of EducationField with respect to Attrition=Yes')
        
        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'education_field_vs_attrition_yes.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

        # Plotting the pie chart for EducationField with respect to Attrition=No
        plt.figure(figsize=(8, 8))
        plt.pie(percentage_no, labels=labels, autopct='%.1f%%', startangle=140, colors=sns.color_palette('Set2'))
        plt.title('Distribution of EducationField with respect to Attrition=No')
        
        # Save the plot before displaying or closing it
        plt.savefig(os.path.join(self.plot_save_dir, 'education_field_vs_attrition_no.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

# Function Call
education_field_plotter = EducationFieldPlot(attrition.data)
education_field_plotter.plot_education_field_distribution()
education_field_plotter.plot_education_field_vs_attrition()


# ##### Employee Satisfaction Analysis

# In[7]:


class SatisfactionAnalysis:
    """
    A class to perform satisfaction analysis with respect to attrition.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
        features (list): List of features to analyze with respect to attrition.
    """
    def __init__(self, data, features):
        """
        Initialize SatisfactionAnalysis object with input data and features.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
            features (list): List of features to analyze with respect to attrition.
        """
        self.data = data
        self.features = features
        self.plot_save_dir = "EDA_PLOTS"  # Directory to save plots
        
        # Create the directory if it doesn't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def plot_attrition_by_feature(self, feature):
        """
        Plot a 100% stacked bar chart of attrition by the specified feature.

        Parameters:
            feature (str): The feature to analyze with respect to attrition.

        Returns:
            None
        """
        # Map satisfaction levels to labels
        satisfaction_labels = {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        }
        
        # Replace satisfaction levels with labels
        self.data[feature] = self.data[feature].map(satisfaction_labels)
        
        # Calculate the proportion of attrition (Yes and No) for each category of the feature
        attrition_proportion = pd.crosstab(self.data[feature], self.data['Attrition'], normalize='index') * 100
        
        # Sort the data based on satisfaction labels
        attrition_proportion = attrition_proportion.reindex(satisfaction_labels.values())

        # Create a 100% stacked bar chart
        ax = attrition_proportion.plot(kind='bar', stacked=True, figsize=(10, 10), color=['#FF9999', '#66B3FF'], alpha=0.8)

        # Set the title and labels
        plt.title(f'100% Stacked Bar Chart of Attrition with respect to {feature}')
        plt.xlabel(feature)
        plt.ylabel('Percentage')

        # Annotate each section of the bar with its corresponding percentage
        for container in ax.containers:
            # Iterate through each bar in the container
            for bar in container:
                # Get the height of the bar (percentage value)
                height = bar.get_height()

                # If the height is greater than 0, annotate the bar with the percentage
                if height > 0:
                    # Calculate the x and y positions for the annotation
                    x_pos = bar.get_x() + bar.get_width() / 2
                    y_pos = bar.get_y() + height / 2

                    # Annotate the bar with the percentage (formatted to 1 decimal place)
                    ax.text(x_pos, y_pos, f'{height:.1f}%', ha='center', va='center', fontsize=10, color='black')

        # Save the plot before closing it
        plt.savefig(os.path.join(self.plot_save_dir, f'attrition_by_{feature}.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

# Function Call
features = ['WorkLifeBalance', 'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']
satisfaction_analyzer = SatisfactionAnalysis(attrition.data, features)
for feature in features:
    satisfaction_analyzer.plot_attrition_by_feature(feature)


# ##### Monthly Income Analysis

# In[8]:


class MonthlyIncomeAnalysis:
    """
    A class to perform analysis on monthly income.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """
    def __init__(self, data):
        """
        Initialize MonthlyIncomeAnalysis object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data
        self.plot_save_dir = "EDA_Plots"  # Directory to save plots
        self.stats_save_dir = "EDA_Plots"  # Directory to save summary statistics
        
        # Create the directories if they don't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)
        os.makedirs(self.stats_save_dir, exist_ok=True)

    def plot_monthly_income_boxplot(self):
        """
        Plot a box plot of MonthlyIncome with respect to Attrition.

        Returns:
            None
        """
        # Plotting a box plot of MonthlyIncome with respect to Attrition
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.data, x='Attrition', y='MonthlyIncome', palette='Set2')
        plt.title('Box Plot of MonthlyIncome with respect to Attrition')
        plt.xlabel('Attrition')
        plt.ylabel('Monthly Income')

        # Save the plot
        plt.savefig(os.path.join(self.plot_save_dir, 'monthly_income_boxplot.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def plot_monthly_income_by_marital_status(self):
        """
        Plot a box plot of MonthlyIncome with respect to MaritalStatus and Attrition.

        Returns:
            None
        """
        # Create a box plot of MonthlyIncome with respect to MaritalStatus and Attrition
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=self.data, x='MaritalStatus', y='MonthlyIncome', hue='Attrition', palette='Set2')

        # Set plot title, x-axis label, and y-axis label
        plt.title('Box Plot of MonthlyIncome with respect to MaritalStatus and Attrition')
        plt.xlabel('Marital Status')
        plt.ylabel('Monthly Income')

        # Save the plot
        plt.savefig(os.path.join(self.plot_save_dir, 'monthly_income_by_marital_status.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def plot_monthly_income_by_job_level(self):
        """
        Plot a box plot of MonthlyIncome with respect to JobLevel.

        Returns:
            None
        """
        # Plot Monthly Income vs Job Level
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.data, x='JobLevel', y='MonthlyIncome', palette='Set2')
        plt.title('Monthly Income vs Job Level')
        plt.xlabel('Job Level')
        plt.ylabel('Monthly Income')
        
        # Save the plot
        plt.savefig(os.path.join(self.plot_save_dir, 'monthly_income_by_job_level.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def calculate_summary_statistics(self):
        """
        Calculate summary statistics (median, mean, quartiles, IQR, and standard deviation) for MonthlyIncome.

        Returns:
            DataFrame: DataFrame containing summary statistics.
        """
        # Group data by Attrition
        grouped_data = self.data.groupby('Attrition')['MonthlyIncome']
        
        # Calculate summary statistics for each group
        summary_stats = grouped_data.describe()
        
        # Save summary statistics as CSV
        summary_stats.to_csv(os.path.join(self.stats_save_dir, 'monthly_income_summary_stats.csv'))
        
        return summary_stats

# Function Call
monthly_income_analyzer = MonthlyIncomeAnalysis(attrition.data)
monthly_income_analyzer.plot_monthly_income_boxplot()
monthly_income_analyzer.plot_monthly_income_by_marital_status()
monthly_income_analyzer.plot_monthly_income_by_job_level()

# Calculate and save summary statistics
summary_stats = monthly_income_analyzer.calculate_summary_statistics()


# ##### Travelling Analysis

# In[9]:


class TravelAttritionAnalysis:
    """
    A class to perform analysis on travel frequency and attrition.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """
    def __init__(self, data):
        """
        Initialize TravelAttritionAnalysis object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data
        self.plot_save_dir = "EDA_PLOTS"  # Directory to save plots
        
        # Create the directory if it doesn't exist
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def plot_travel_vs_attrition_count(self):
        """
        Plot a countplot of BusinessTravel with respect to Attrition.

        Returns:
            None
        """
        # Plotting the countplot of BusinessTravel with respect to Attrition
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=self.data, x='BusinessTravel', hue='Attrition', palette='Set2')
        plt.title('Travel Frequency vs Attrition')
        plt.xlabel('Business Travel')
        plt.ylabel('Count')

        # Annotating each bar with its count value
        for p in ax.patches:
            count = p.get_height()  # Get the height of the bar (count value)
            # Annotate each bar with its count value, positioning the text slightly above the top of the bar
            ax.annotate(f'{int(count)}', (p.get_x() + p.get_width() / 2., count),  # Position text at the top of the bar
                        ha='center', va='bottom',  # Center and bottom alignment
                        xytext=(0, 2),  # Offset the text by 2 points upward
                        textcoords='offset points')

        # Save the plot
        plt.savefig(os.path.join(self.plot_save_dir, 'travel_vs_attrition_count.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

    def plot_100_percent_stacked_travel_vs_attrition(self):
        """
        Plot a 100% stacked bar plot of BusinessTravel with respect to Attrition.

        Returns:
            None
        """
        # Group the data by BusinessTravel and Attrition and count the number of occurrences
        grouped_data = self.data.groupby(['BusinessTravel', 'Attrition']).size().reset_index(name='count')

        # Calculate the total count for each BusinessTravel category
        total_count = grouped_data.groupby('BusinessTravel')['count'].transform('sum')

        # Calculate the proportion of each category within each BusinessTravel group
        grouped_data['proportion'] = grouped_data['count'] / total_count * 100

        # Pivot the data to create a wide-form DataFrame for 100% stacked bar plot
        pivoted_data = grouped_data.pivot(index='BusinessTravel', columns='Attrition', values='proportion').fillna(0)

        # Plotting the 100% stacked bar plot of BusinessTravel with respect to Attrition
        plt.figure(figsize=(10, 5))
        ax = pivoted_data.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], edgecolor='k')

        plt.title('100% Stacked Bar Plot of Travel Frequency with respect to Attrition')
        plt.xlabel('Business Travel')
        plt.ylabel('Percentage')

        # Annotating each bar with its percentage value
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.annotate(f'{int(height)}%', (x + width / 2, y + height / 2), ha='center', va='center')

        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.plot_save_dir, '100_percent_stacked_travel_vs_attrition.png'))
        
        # Close the plot to prevent it from being displayed
        plt.close()

# Function Call
travel_attrition_analyzer = TravelAttritionAnalysis(attrition.data)
travel_attrition_analyzer.plot_travel_vs_attrition_count()
travel_attrition_analyzer.plot_100_percent_stacked_travel_vs_attrition()


# ##### Work Exp. Analysis

# In[10]:


class CountPlotter:
    """
    A class to create count plots with annotations.

    Attributes:
        data (DataFrame): Input DataFrame containing relevant data.
    """
    def __init__(self, data):
        """
        Initialize CountPlotter object with input data.

        Parameters:
            data (DataFrame): Input DataFrame containing relevant data.
        """
        self.data = data

    def create_count_plot(self, x, hue, title, xlabel, ylabel):
        """
        Create a count plot with annotations and save it to EDA_PLOTS folder.

        Parameters:
            x (str): Column name for the x-axis.
            hue (str): Column name for the hue (grouping variable).
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.

        Returns:
            None
        """
        # Plotting the count plot
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=self.data, x=x, hue=hue, palette='Set2')

        # Set plot title and labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Annotating each bar with its count value
        for p in ax.patches:
            count = p.get_height()  # Get the height of the bar (count value)
            # Annotate each bar with its count value, positioning the text slightly above the top of the bar
            ax.annotate(f'{int(count)}', (p.get_x() + p.get_width() / 2., count),  # Position text at the top of the bar
                        ha='center', va='bottom',  # Center and bottom alignment
                        xytext=(0, 2),  # Offset the text by 2 points upward
                        textcoords='offset points')

        # Save the plot to EDA_PLOTS folder
        if not os.path.exists("EDA_PLOTS"):
            os.makedirs("EDA_PLOTS")
        plt.savefig(f"EDA_PLOTS/{title.replace(' ', '_')}.png")
        plt.close()

# Function Call
count_plotter = CountPlotter(attrition.data)
count_plotter.create_count_plot(
    x='YearsInCurrentRole',
    hue='Attrition',
    title='Count Plot of Years in Current Role with respect to Attrition',
    xlabel='Years in Current Role',
    ylabel='Count'
)
count_plotter.create_count_plot(
    x='JobLevel',
    hue='Attrition',
    title='Count Plot of Job Level with respect to Attrition',
    xlabel='Job Level',
    ylabel='Count'
)
count_plotter.create_count_plot(
    x='YearsWithCurrManager',
    hue='Attrition',
    title='Count Plot of Years With Current Manager with respect to Attrition',
    xlabel='Years With Current Manager',
    ylabel='Count'
)
count_plotter.create_count_plot(
    x='YearsAtCompany',
    hue='Attrition',
    title='Count Plot of Years At Company with respect to Attrition',
    xlabel='Years At Company',
    ylabel='Count'
)
count_plotter.create_count_plot(
    x='YearsSinceLastPromotion',
    hue='Attrition',
    title='Count Plot of Years Since Last Promotion with respect to Attrition',
    xlabel='Years Since Last Promotion',
    ylabel='Count'
)

