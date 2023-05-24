---
title: "Comprehensive Medicare Data Analysis for Insurance Companies"
excerpt: "In this study, the team leveraged Big Data techniques to analyze the CMS Medicare data and provide valuable insights for insurance companies to optimize their policies and coverage options. By effectively handling the large-scale dataset, we were able to uncover patterns in healthcare service utilization, payment dynamics, and drug claims among insured policyholders, leading to actionable recommendations for improved healthcare delivery.<br /><img src='/images/cms/1.png'>"
date: May 15, 2023
collection: portfolio
---
"Under Construction" 

<img src='/images/cms/1.png'>

<a id='ExecSum'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">EXECUTIVE SUMMARY</h1>


In an era where healthcare costs are soaring, and the demand for efficient and high-quality care is increasing, understanding the patterns of healthcare service utilization and associated costs is of paramount importance. This study is motivated by the desire to identify opportunities for insurance companies to optimize their policies, expand coverage options, and ultimately, deliver better healthcare to Medicare beneficiaries.

To address the research question **"How can insurance companies optimize their policies and coverage options based on the analysis of CMS Medicare data?"**, the team embarked on a comprehensive exploration of the CMS Data by Provider and Service and by Provider and Drug. This endeavor aimed to uncover valuable insights into healthcare service utilization patterns, payment dynamics, submitted charges, and drug claims among insured policyholders. By navigating the complexities of this data, the team sought to provide solutions that would enable insurance companies to optimize their policies and coverage options effectively.

The methodology employed by the team can be summarized into three key sections. It began with data collection, followed by preprocessing of publicly available Centers for Medicare & Medicaid Services (CMS) datasets, and concluded with data analysis involving descriptive analytics and visualization. The team conducted a comprehensive descriptive analysis and visualization of the data, breaking it down into several sections from the distribution of service providers to the service and drug claims across different age groups. This comprehensive analysis enabled the team to delve into numerous aspects of the Medicare system, such as the role of different healthcare specialties, claims and payments, geographical differences in service provision and pricing, the impact of age factor, and the distribution and utilization of prescription drugs.

The analysis revealed significant patterns in healthcare service utilization, payment dynamics, submitted charges, and drug claims among insured policy holders. Such results underscored the significant role of primary care services, particularly `Internal Medicine` and `Family Practice`, in the healthcare delivery system. It also highlighted the high dependency on Medicare among the elderly population, with a notably higher number of claims in the `65 and above` age group. The high volume of claims for certain specialties, such as `Cardiology`, `Psychiatry`, `Neurology`, and `Nephrology`, underscored the prevalence of age-related conditions in this group.

A key finding was the dominance of `individual providers` over organizations in Medicare claims, suggesting a fragmented healthcare market within Medicare that caters to diverse patient needs. The distribution of service locations also pointed to a preference among patients for care in `offices` compared to `facilities`, aligning with the dominance of `individual providers` in Medicare.

The data further revealed that `US providers` and `non-US providers` differed significantly in their submitted charges for services, although the average Medicare payment did not significantly vary between these two groups.

These findings lay the foundation for actionable recommendations that can lead to more profitable outcomes for insurance companies. Such findings helped the team solidify and summarize the recommendations which were categorized into immediate implementations and future work. For immediate implementation, the team suggested revising primary care coverage and premiums, adjusting reimbursement for `Nurse Practitioners` and `Physician Assistants`, standardizing service costs across different geographic locations, and incentivizing non-participating providers to capture more market share.

Future work recommendations included introducing specialty-specific premiums, revising prescription drug coverage, developing a risk stratification model for beneficiaries, investing in data infrastructure, and implementing targeted health promotion programs.

While the analysis provides valuable insights, it is important to acknowledge the assumptions and limitations. These include the assumption of data completeness, accuracy, and consistent categorization. However, limitations such as temporal constraints, the absence of clinical data and detailed provider information, and geographical limitations should be considered. Addressing these limitations in future studies can further enhance the understanding of healthcare utilization and costs within the Medicare system.

<a id='TContents'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">TABLE OF CONTENTS</h1>

[EXECUTIVE SUMMARY](#ExecSum) <br> 
[LIBRARIES & FUNCTIONS](#Libraries) <br> 
[1. PROBLEM STATEMENT](#Problem) <br> 
[2. MOTIVATION](#Motivation) <br> 
[3. METHODOLOGY](#Methodology) <br> 
[4. DATA COLLECTION](#Collection) <br> 
- [4.1 Data Source](#Source) 
- [4.2 Data Description](#Description) 
- [4.3 Data Collection](#Collect) <br> 

[5. DATA PREPROCESSING AND EDA](#Prepare) <br> 
[6. DATA ANALYSIS & VISUALIZATION](#Analytics) <br> 
[7. RESULTS AND DISCUSSION](#Results) <br>
[8. CONCLUSION](#Conclusion) <br> 
[9. RECOMMENDATION](#Recommendation) <br> 
- [9.1 Limitations of the Study](#Limitations)
- [9.2 Recommendation](#Recommendation2)
- [9.3 Future Work](#FutureWork) <br> 

[REFERENCES](#References) <br> 

<h2 style="color:#0d2599; font-size:26px;">List of Tables</h2>

- Table 1. Methodology Details 
- Table 2. Data Description of Features Used: Data by Provider and Service
- Table 3. Data Description of Features Used: Data by Provider and Drug
- Table 4. Cleaned DataFrame: Data by Provider and Service
- Table 5. Cleaned DataFrame: Data by Provider and Drug
- Table 6. Statistics: Data by Provider and Service
- Table 7. Statistics: Data by Provider and Drug
- Table 8. Top 15 Specialties by % Medicare Covered
- Table 9. Bottom 10 Specialties by % Medicare Covered
- Table 10. Claims, Payments, and Beneficiary count between US and Non-US

<h2 style="color:#0d2599; font-size:26px;">List of Figures</h2>

- Figure 1. Methodology
- Figure 2. Plot Number of Organizations and Individuals
- Figure 3. Plot Distribution of Place of Service
- Figure 4. Plot Medicare Participation Distribution
- Figure 5. Plot Top 10 Provider Specialties
- Figure 6. Plot Pareto Chart by Total Submitted Charges
- Figure 7. Plot Pareto Chart by Total Medicare Payment Amount
- Figure 8. Plot by % Medicare Covered
- Figure 9. Plot Top 10 Non-US Countries by Total Submitted Charges
- Figure 10. Plot Top 10 US Cities by Total Submitted Charges
- Figure 11. Plot Top 10 Drug by Total Claims
- Figure 12. Plot Distribution of Total Claims by Age Group
- Figure 13. Plot Total Claims of Top 10 Specialties by Age Group

<a id='Libraries'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">LIBRARIES AND FUNCTIONS</h1>

This section establishes the initial steps for the subsequent analysis by importing the necessary libraries and defining the required functions. These includes global libraries and helper functions, which serve as the foundation for the analysis process.

<h2 style="color:#0d2599; font-size:26px;">Global Libraries</h2>

```python
# Import Libraries and modules
from pyspark.sql import SparkSession
import pyspark.pandas as ps
from pyspark.sql.functions import isnan, when, count, col, avg, desc
from pyspark.sql.functions import sum, format_number, expr
from pyspark.sql.types import (IntegerType, TimestampType, DoubleType,
                               StringType, LongType, StructType, StructField)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import multiprocessing
import warnings
from IPython.display import HTML
from IPython.display import display, display_html


# Global settings
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# Hide warnings
warnings.filterwarnings("ignore")


# Global variables
c_blue = '#0d2599'
c_lightblue= '#e0eaf8'
c_green = '#42e288'
c_black = '#212121'
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'] # Custom color palette
c_blues = plt.get_cmap('Blues')(np.linspace(0.15, 0.60, 10))


# Filepaths in jojie
filepath_svc = "/mnt/data/public/cms-gov/" \
               "Medicare_Provider_Util_Payment_PUF_CY201[3-5]/*.txt"
filepath_drug = "/mnt/data/public/cms-gov/" \
                "PartD_Prescriber_PUF_NPI_DRUG_1[3-6]/*.txt"
```

<h2 style="color:#0d2599; font-size:26px;">Helper Functions</h2>

```python
# Define Global Helper Functions
def preview_df(data_label, df, display_df=True, display_nulls=False):
    """Displays the data label, number of data points and features, columns 
    with null values, and preview of the DataFrame.
    """
    # Calculate the number of data points and features
    num_data_points = df.count()
    num_features = len(df.columns)

    # Iterate over all columns and check for null values
    null_columns = []
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            null_columns.append((column, null_count))

    # Create a new DataFrame with columns and their null counts
    null_columns_df = spark.createDataFrame(null_columns, 
                                            ["Column", "Null Count"])

    # Display the results
    display(HTML(f'<b>Data Label:</b> {data_label}'))
    display(HTML(f'<b>Number of Data Points:</b> {num_data_points}'))
    display(HTML(f'<b>Number of Features:</b> {num_features}'))
    
    # If display_nulls=True, display DataFrame with null values & count
    if display_nulls:    
        display(HTML(f'<b>Columns with null values:</b>'))
        display(null_columns_df.toPandas())
    
    # If display_df=True, display 3 sample rows of the current dataframe
    if display_df:
        display(HTML(f'<b>Preview of the dataframe:</b>'))
        display(df.limit(3).toPandas())


def preview_service(data_label, df, display_df=True, display_nulls=False):
    """Displays the data label, number of data points and features, columns 
    with null values, and preview of the DataFrame.
    """
    # Calculate the number of data points and features
    num_data_points = df.count()
    num_features = len(df.columns)

    # Iterate over all columns and check for null values
    null_columns = []
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            null_columns.append((column, null_count))

    # Create a new DataFrame with columns and their null counts
    null_columns_df = spark.createDataFrame(null_columns, 
                                            ["Column", "Null Count"])

    # Display the results
    display(HTML(f'<b>Data Label:</b> {data_label}'))
    display(HTML(f'<b>Number of Data Points:</b> {num_data_points}'))
    display(HTML(f'<b>Number of Features:</b> {num_features}'))
    
    # If display_nulls=True, display DataFrame with null values & count
    if display_nulls:    
        display(HTML(f'<b>Columns with null values:</b>'))
        display(null_columns_df.toPandas())
    
    # If display_df=True, display 3 sample rows of the current dataframe
    if display_df:
        display(HTML(f'<b>Preview of the dataframe:</b>'))
        display(HTML(f'<left><span style="color:gray; '
                     f'font-size:16px"><b>Table 4. </b>Cleaned DataFrame: '
                     f'Data by Provider and Service</span></left>'))
        display(df.limit(3).toPandas())


def preview_drug(data_label, df, display_df=True, display_nulls=False):
    """Displays the data label, number of data points and features, columns 
    with null values, and preview of the DataFrame.
    """
    # Calculate the number of data points and features
    num_data_points = df.count()
    num_features = len(df.columns)

    # Iterate over all columns and check for null values
    null_columns = []
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            null_columns.append((column, null_count))

    # Create a new DataFrame with columns and their null counts
    null_columns_df = spark.createDataFrame(null_columns, 
                                            ["Column", "Null Count"])

    # Display the results
    display(HTML(f'<b>Data Label:</b> {data_label}'))
    display(HTML(f'<b>Number of Data Points:</b> {num_data_points}'))
    display(HTML(f'<b>Number of Features:</b> {num_features}'))
    
    # If display_nulls=True, display DataFrame with null values & count
    if display_nulls:    
        display(HTML(f'<b>Columns with null values:</b>'))
        display(null_columns_df.toPandas())
    
    # If display_df=True, display 3 sample rows of the current dataframe
    if display_df:
        display(HTML(f'<b>Preview of the dataframe:</b>'))
        display(HTML(f'<left><span style="color:gray; '
                     f'font-size:16px"><b>Table 5. </b>Cleaned DataFrame: '
                     f'Data by Provider and Drug</span></left>'))
        display(df.limit(3).toPandas())


def drop_nulls(df):
    """Displays the rows with 'NPI'==1, drops the rows with 'NPI'=1,
    and displays a sample of 3 rows from the cleaned DataFrame.
    """
    # Display rows with 'NPI'==1
    display(HTML(f'<b>Dropping rows with NPI=1 in provider_service df:</b>'))
    npi_equals_1_df = df.filter(col("NPI") == 1)
    display(npi_equals_1_df.toPandas())
    display(HTML(f'<b>Dropping complete.</b>'))

    # Drop rows with 'NPI'=1
    cleaned_df = df.filter(col("NPI") != 1)

    return cleaned_df


def drop_columns(data_label, df, columns_to_drop):
    """Drops the given columns from the DataFrame."""
    cleaned_df = df.drop(*columns_to_drop)
    display(HTML(f'<b>Data Label:</b> {data_label}'))
    display(HTML(f'<b>Dropped columns: {columns_to_drop}</b>'))
    return cleaned_df


def clean_names_service(data_label, df):
    """Corrects the first and last names of NPI#1912175340."""
    display(HTML(f'<b>Data Label:</b> {data_label}'))
    display(HTML(f'<b>Correcting the first & last name of '
                 f'NPI#1912175340 ...</b>'))
    df = df.withColumn(
        "NPPES_PROVIDER_LAST_ORG_NAME",
        when(
            col("NPPES_PROVIDER_LAST_ORG_NAME") == "&H'S)U",
            "HSU "
        ).otherwise(col("NPPES_PROVIDER_LAST_ORG_NAME"))
    ).withColumn(
        "NPPES_PROVIDER_FIRST_NAME",
        when(
            col("NPPES_PROVIDER_FIRST_NAME") == "&E'K:(A:I",
            "EKAI"
        ).when(
            col("NPPES_PROVIDER_FIRST_NAME") == "E-KAI",
            "EKAI"
        ).otherwise(col("NPPES_PROVIDER_FIRST_NAME"))
    )
    display(HTML(f'<b>Task completed.</b>'))
    return df


def clean_names_drug(data_label, df):
    """Corrects the first and last names of NPI#1912175340."""
    display(HTML(f'<b>Data Label:</b> {data_label}'))
    display(HTML(f'<b>Correcting the first and last name '
                 f'of NPI#1912175340 ...</b>'))
    df = df.withColumn(
        "nppes_provider_last_org_name",
        when(
            col("nppes_provider_last_org_name") == "&H'S)U",
            "HSU "
        ).otherwise(col("nppes_provider_last_org_name"))
    ).withColumn(
        "nppes_provider_first_name",
        when(
            col("nppes_provider_first_name") == "&E'K:(A:I",
            "EKAI"
        ).when(
            col("nppes_provider_first_name") == "E-KAI",
            "EKAI"
        ).otherwise(col("nppes_provider_first_name"))
    )
    display(HTML(f'<b>Task completed.</b>'))    
    return df


def plot_by_entity(provider_service):
    """Plots distribution of Medicare Providers based on Entity Code."""
    entity_counts = (provider_service
                     .groupby('NPPES_ENTITY_CODE')
                     .count()
                     .toPandas())

    # Dictionary to map abbreviations to spelled-out labels
    entity_labels = {
        'O': 'Organization',
        'I': 'Individual'}
    colors = ['#ff9999', '#66b3ff']

    # Replace abbreviations with spelled-out labels
    entity_counts['NPPES_ENTITY_CODE'] = (entity_counts['NPPES_ENTITY_CODE']
                                          .map(entity_labels))

    # Plotting a pie chart with custom colors and labels
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(entity_counts['count'], labels=entity_counts['NPPES_ENTITY_CODE'], 
           autopct='%1.1f%%', colors=colors, textprops={'fontsize': 16})
    ax.set_title(f'Medicare Providers: '
                 f'Individuals Outnumbering Organizations', 
                 fontdict={'fontsize': 20, 
                           'fontweight': 'bold', 
                           'color':'#0d2599'})
    ax.axis('equal')

    plt.tight_layout()
    plt.show()


def plot_age_group_distribution(provider_drug):
    """Plots the distribution of total claims by age group in a pie chart."""
    data = (provider_drug.groupby(['ge65_suppress_flag'])
            .agg(count('*').alias('count'))
            .withColumn('Age Group', 
                        when(col('ge65_suppress_flag') == '*',
                             'Age below 65').otherwise('Age 65 and above'))
            .drop('ge65_suppress_flag')
            .groupby('Age Group')
            .agg(sum('count').alias('Total Claims'))
            .toPandas())

    # Extract the necessary data for the pie chart
    labels = data['Age Group']
    sizes = data['Total Claims']
    colors = ['#ff9999', '#66b3ff']

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
          textprops={'fontsize': 16})

    # Set aspect ratio to be equal so that pie is drawn as a circle
    ax.axis('equal')

    # Set title
    ax.set_title('Distribution of Total Claims by Age Group', 
                 fontsize=20, fontweight='bold', color='#0d2599')

    # Display the chart
    plt.show()


def millions_formatter(x, pos):
    """Formatter function to display axis values in millions."""
    return f'{x/1e6:.1f}M'


def plot_place_of_service(df):
    """Plots the distribution of PLACE_OF_SERVICE based on count."""
    counts = df.groupby('PLACE_OF_SERVICE').count().toPandas()

    # Dictionary to map abbreviations to spelled-out labels
    labels = {'F': 'Facility', 'O': 'Others (Offices)'}

    # Bar plot with custom colors and fontsize
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = (ax.bar([labels.get(code, code) for code in 
                    counts['PLACE_OF_SERVICE']], 
                  counts['count'], 
                  color=['#ff9999', '#66b3ff']))
    ax.set_xlabel('Place of Service', fontsize=20)
    ax.set_ylabel('Count (in millions)', fontsize=20)
    ax.set_title('Medicare Service Locations: Offices Lead over Facilities', 
                 fontsize=20, fontweight='bold', color='#0d2599')
    ax.tick_params(axis='both', labelsize=16)

    # Format y-axis values in millions
    formatter = FuncFormatter(millions_formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Place labels inside the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height/1e6:.1f}M', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points", 
                    ha='center', 
                    va='bottom')
    plt.show()


def plot_top_provider_specialties(provider_service):
    """Plots the top 10 provider specialties based on counts."""
    top_specialties = (provider_service
                       .groupBy('PROVIDER_TYPE')
                       .agg(count('*').alias('count'))
                       .sort('count', ascending=False)
                       .limit(10)
                       .toPandas())

    # Create the vertical bar plot with reversed order
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_specialties['PROVIDER_TYPE'][::-1],  # Reverse the order
            top_specialties['count'][::-1],  # Reverse the order of counts
            color=['#e0eaf8']*7 + ['#66b3ff']*3)

    ax.set_xlabel('Count (Millions)', fontsize=16)
    ax.set_ylabel('Provider Specialty', fontsize=16)
    ax.set_title('Top 10 Provider Specialties', fontsize=20, 
                 fontweight='bold', color='#0d2599')
    ax.tick_params(axis='y', rotation=0)

    # Format the y-axis labels in millions
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, 
                                                      pos: f'{x/1e6:.1f}M'))

    plt.tight_layout()
    plt.show()


def plot_by_medicare_participation(df):
    """Plots the distribution of Medicare Providers based on 
    Medicare Participation Indicator.
    """
    df.select('MEDICARE_PARTICIPATION_INDICATOR', 'NPI')
    counts = (df
              .groupBy('MEDICARE_PARTICIPATION_INDICATOR')
              .agg(count('*').alias('count'))
              .toPandas())

    # Dictionary to map abbreviations to spelled-out labels
    labels = {'Y': 'Yes', 'N': 'No'}

    # Pie chart with custom colors and fontsize
    fig, ax = plt.subplots(figsize=(8, 6))
    (ax.pie(counts['count'], 
           labels=[labels.get(code, code) 
                   for code in counts['MEDICARE_PARTICIPATION_INDICATOR']],
            autopct='%1.2f%%', 
            colors=['#66b3ff', '#ff9999'], 
            textprops={'fontsize': 16}))
    ax.set_title('Medicare Providers: Vast Majority in Participation', 
                 fontsize=20, 
                 fontweight='bold',
                 color='#0d2599')
    plt.tight_layout()
    plt.show()


def plot_pareto_specialties_by_submitted_charges(provider_service):
    """Plots the top provider specialties based on the average submitted 
    charge amount as a Pareto chart.
    """
    top_specialties = (provider_service
                       .groupBy('PROVIDER_TYPE')
                       .agg({'average_submitted_chrg_amt': 'sum'})
                       .withColumnRenamed('sum(average_submitted_chrg_amt)', 
                                          'average_submitted_chrg_amt')
                       .sort('average_submitted_chrg_amt', ascending=False)
                       .limit(22)
                       .toPandas())

    # Total overall sum
    total_sum = (provider_service
                 .agg({'average_submitted_chrg_amt': 'sum'})
                 .collect()[0][0])

    # Calculate cumulative percentage
    top_specialties['cumulative_percentage'] = (
        (top_specialties['average_submitted_chrg_amt'] / total_sum).cumsum())

    # Group the difference as "Others"
    others_sum = total_sum-top_specialties['average_submitted_chrg_amt'].sum()
    others_row = pd.DataFrame({'PROVIDER_TYPE': ['Others'],
                               'average_submitted_chrg_amt': [others_sum]})
    top_specialties = pd.concat([top_specialties, others_row])

    # Create the Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot the bars
    ax1.bar(top_specialties['PROVIDER_TYPE'],
            top_specialties['average_submitted_chrg_amt'], 
            color=['#66b3ff']*22 + ['#e0eaf8']*1)

    ax1.set_xlabel('Provider Specialty', fontsize=16, color='#0d2599')
    ax1.set_ylabel('Total Submitted Charge Amount (Millions)', 
                   fontsize=16, 
                   fontweight='bold',
                   color='#66b3ff')
    ax1.set_title("80/20 Rule: Customer Specialty Needs thru Submitted Costs",
                  fontsize=20,
                  fontweight='bold',
                  color='#0d2599')

    # Format the ax1 y-axis labels in millions
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Create the secondary y-axis for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(top_specialties['PROVIDER_TYPE'],
             top_specialties['cumulative_percentage'],
             color='#ff0000', marker='o')

    ax2.set_ylabel('Cumulative Percentage', 
                   fontsize=16, 
                   fontweight='bold',
                   color='#ff0000')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim([0, 1])  # Set the y-axis range from 0 to 100%

    ax1.tick_params(axis='y', labelcolor='#66b3ff')
    ax2.tick_params(axis='y', labelcolor='#ff0000')

    # Rotate x-axis labels for better visibility
    ax1.set_xticklabels(top_specialties['PROVIDER_TYPE'], 
                        rotation=45, 
                        ha='right', 
                        color='#0d2599')

    # Annotate the total overall sum
    ax1.text(0.5, 0.35, f"Total Overall Submitted Charges: ${total_sum:,.2f}",
             transform=ax1.transAxes, color='#0d2599', 
             ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_pareto_specialties_by_payment_charges(provider_service):
    """Plots the top provider specialties based on the total 
    Medicare payment amount as a Pareto chart.
    """
    top_specialties = (provider_service
                       .groupBy('PROVIDER_TYPE')
                       .agg({'AVERAGE_MEDICARE_PAYMENT_AMT': 'sum'})
                       .withColumnRenamed('sum(AVERAGE_MEDICARE_PAYMENT_AMT)', 
                                          'AVERAGE_MEDICARE_PAYMENT_AMT')
                       .sort('AVERAGE_MEDICARE_PAYMENT_AMT', ascending=False)
                       .limit(23)
                       .toPandas())

    # Total overall sum
    total_sum = (provider_service
                 .agg({'AVERAGE_MEDICARE_PAYMENT_AMT': 'sum'})
                 .collect()[0][0])

    # Calculate cumulative percentage
    top_specialties['cumulative_percentage'] = (
        (top_specialties['AVERAGE_MEDICARE_PAYMENT_AMT']/total_sum).cumsum())

    # Group the difference as "Others"
    others_sum = (total_sum-top_specialties['AVERAGE_MEDICARE_PAYMENT_AMT']
                  .sum())
    others_row = pd.DataFrame({'PROVIDER_TYPE': ['Others'],
                               'AVERAGE_MEDICARE_PAYMENT_AMT': [others_sum]})
    top_specialties = pd.concat([top_specialties, others_row])

    # Create the Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot the bars
    ax1.bar(top_specialties['PROVIDER_TYPE'],
            top_specialties['AVERAGE_MEDICARE_PAYMENT_AMT'], 
            color=['#66b3ff']*23 + ['#e0eaf8']*1)

    ax1.set_xlabel('Provider Specialty', fontsize=16, color='#0d2599')
    ax1.set_ylabel('Total Medicare Payment Amount (Millions)', 
                   fontsize=16, 
                   fontweight='bold',
                   color='#66b3ff')
    ax1.set_title(f"80/20 Rule: 23 out of 91 Specialties Leading "
                  f"Medicare Payments",
                  fontsize=20,
                  fontweight='bold',
                  color='#0d2599')

    # Format the ax1 y-axis labels in millions
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Create the secondary y-axis for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(top_specialties['PROVIDER_TYPE'],
             top_specialties['cumulative_percentage'],
             color='#ff0000', marker='o')

    ax2.set_ylabel('Cumulative Percentage', 
                   fontsize=16, 
                   fontweight='bold',
                   color='#ff0000')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim([0, 1])  # Set the y-axis range from 0 to 100%

    ax1.tick_params(axis='y', labelcolor='#66b3ff')
    ax2.tick_params(axis='y', labelcolor='#ff0000')

    # Rotate x-axis labels for better visibility
    ax1.set_xticklabels(top_specialties['PROVIDER_TYPE'], 
                        rotation=45, 
                        ha='right', 
                        color='#0d2599')

    # Annotate the total overall sum
    ax1.text(0.5, 0.4, f"Total Medicard Paid Charges: ${total_sum:,.2f}"
             f"\n16% of Total Submitted Charges",
             transform=ax1.transAxes, color='#0d2599', ha='center', 
             va='center', fontsize=12)

    plt.tight_layout()
    plt.show()


def medicare_covered(provider_service):
    """Calculates the sum of AVERAGE_SUBMITTED_CHRG_AMT and 
    AVERAGE_MEDICARE_PAYMENT_AMT, as well as the % medicare covered,
    grouped by PROVIDER_TYPE.
    """
    medicare_covered_df = (provider_service
                            .groupBy('PROVIDER_TYPE')
                            .agg(sum('AVERAGE_SUBMITTED_CHRG_AMT')
                                 .alias('total_submitted_charges'),
                                 sum('AVERAGE_MEDICARE_PAYMENT_AMT')
                                 .alias('total_medicare_payment'))
                            .withColumn('total_submitted_charges', 
                                        col('total_submitted_charges')
                                        .cast('float'))
                            .withColumn('total_medicare_payment', 
                                        col('total_medicare_payment')
                                        .cast('float'))
                            .withColumn('% medicare covered', 
                                        expr(f'total_medicare_payment / '
                                             f'total_submitted_charges*100'))
                            .withColumn('% medicare covered', when(
                                col('total_submitted_charges') != 0,
                                col('% medicare covered')).otherwise(0))
                            .sort('% medicare covered', ascending=False)
                            .withColumn('total_submitted_charges', 
                                        format_number(
                                            'total_submitted_charges', 2))
                            .withColumn('total_medicare_payment', 
                                        format_number(
                                            'total_medicare_payment', 2))
                            .withColumn('% medicare covered', 
                                        format_number(
                                            '% medicare covered', 2)))

    return medicare_covered_df.toPandas()


def calculate_sum_by_provider_type(provider_service):
    """Calculates the sum of AVERAGE_SUBMITTED_CHRG_AMT and 
    AVERAGE_MEDICARE_PAYMENT_AMT grouped by PROVIDER_TYPE.
    """
    medicare_covered = (provider_service
                            .groupBy('PROVIDER_TYPE')
                            .agg(sum('AVERAGE_SUBMITTED_CHRG_AMT')
                                 .alias('total_submitted_charges'),
                                 sum('AVERAGE_MEDICARE_PAYMENT_AMT')
                                 .alias('total_medicare_payment'))
                            .withColumn('total_submitted_charges', 
                                        col('total_submitted_charges')
                                        .cast('float'))
                            .withColumn('total_medicare_payment', 
                                        col('total_medicare_payment')
                                        .cast('float'))
                            .withColumn('% medicare covered', 
                                        expr(f'total_medicare_payment / '
                                             f'total_submitted_charges*100'))
                            .withColumn('% medicare covered', 
                                        when(col(
                                            'total_submitted_charges') != 0,
                                             col('% medicare covered'))
                                        .otherwise(0))
                            .sort('total_medicare_payment', ascending=False)
                            .limit(20))

    return medicare_covered.toPandas()


def plot_medicare_coverage(sum_by_provider_type):
    """Plots the total submitted charges, total Medicare payments, 
    and percentage of Medicare covered by provider type.
    """
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()

    # Plot the overlapping bar chart in ax1
    width = 0.35
    x_pos = np.arange(len(sum_by_provider_type['PROVIDER_TYPE']))

    ax1.bar(x_pos - width/2, 
            sum_by_provider_type['total_submitted_charges'], 
            width, color='#66b3ff', label='Total Submitted Charges')
    ax1.bar(x_pos + width/2, 
            sum_by_provider_type['total_medicare_payment'], 
            width, color='#ff9999', label='Total Medicare Payments')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sum_by_provider_type['PROVIDER_TYPE'], 
                        rotation=45, ha='right')
    ax1.set_xlabel('Provider Type', fontsize=16)
    ax1.set_ylabel('Amount', fontsize=16)
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax1.set_title(f'% Medicare Coverage Gaps Evident in Top 20 '
                  f'Provider Specialties', fontsize=20, 
                  color='#0d2599', fontweight='bold')
    ax1.legend(loc='upper right')

    # Plot the line graph in ax2
    ax2.plot(sum_by_provider_type['PROVIDER_TYPE'], 
             sum_by_provider_type['% medicare covered'], 
             color='#ff0000', marker='o')
    ax2.set_ylabel('% Medicare Covered', fontsize=16, color='#ff0000')
    ax2.set_ylim([0, 100])

    # Update colors of y tickmarks
    ax2.tick_params(axis='y', labelcolor='#ff0000')
    
    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def display_charge_beneficiary_average(data):
    """Displays a Spark DataFrame with the average of 
    'AVERAGE_SUBMITTED_CHRG_AMT', 'AVERAGE_MEDICARE_PAYMENT_AMT', 
    and the average of 'BENE_DAY_SRVC_CNT' grouped by 
    'NPPES_PROVIDER_COUNTRY'.
    """
    return (data.groupby('NPPES_PROVIDER_COUNTRY')
            .agg(avg('AVERAGE_SUBMITTED_CHRG_AMT')
                 .alias('AVERAGE_SUBMITTED_CHRG_AMT'), 
                 avg('AVERAGE_MEDICARE_PAYMENT_AMT')
                 .alias('AVERAGE_MEDICARE_PAYMENT_AMT'),
                 avg('BENE_DAY_SRVC_CNT').alias('BENE_DAY_SRVC_CNT'))
            .withColumn('NPPES_PROVIDER_COUNTRY', 
                        when(col('NPPES_PROVIDER_COUNTRY') != 'US', 'Non-US')
                        .otherwise(col('NPPES_PROVIDER_COUNTRY')))
            .groupby('NPPES_PROVIDER_COUNTRY')
                        .agg(avg('AVERAGE_SUBMITTED_CHRG_AMT')
                             .alias('AVERAGE_SUBMITTED_CHRG_AMT'), 
                             avg('AVERAGE_MEDICARE_PAYMENT_AMT')
                             .alias('AVERAGE_MEDICARE_PAYMENT_AMT'),
                             avg('BENE_DAY_SRVC_CNT')
                             .alias('BENE_DAY_SRVC_CNT'))
            .withColumn('AVERAGE_SUBMITTED_CHRG_AMT', 
                        format_number('AVERAGE_SUBMITTED_CHRG_AMT', 2))
            .withColumn('AVERAGE_MEDICARE_PAYMENT_AMT', 
                        format_number('AVERAGE_MEDICARE_PAYMENT_AMT', 2))
            .toPandas())


def plot_top_non_us_countries_by_submitted_charges(data, n=10):
    """Plots a bar chart of the top N non-US countries based on 
    total submitted charges.
    """
    # Filter out the United States (US) from the data
    data = data.filter(data.NPPES_PROVIDER_COUNTRY != 'US')

    # Group the data by country and calculate the sum of submitted charges
    sum_charges_by_country = (data.groupby('NPPES_PROVIDER_COUNTRY')
                              .agg({'AVERAGE_SUBMITTED_CHRG_AMT': 'sum'})
                              .sort('sum(AVERAGE_SUBMITTED_CHRG_AMT)', 
                                    ascending=False)
                              .limit(n)
                              .toPandas())

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bar chart
    ax.bar(sum_charges_by_country['NPPES_PROVIDER_COUNTRY'], 
           sum_charges_by_country['sum(AVERAGE_SUBMITTED_CHRG_AMT)'], 
           color='#66b3ff')

    # Set the labels and title
    ax.set_xlabel('Country', fontsize=16)
    ax.set_ylabel('Total Submitted Charges', fontsize=16)
    ax.set_title('Top {} Countries by Total Submitted Charges (Excluding US)'
                 .format(n), fontsize=20, color='#0d2599', fontweight='bold')

#     # Format y-axis labels in millions
#     ax.yaxis.set_major_formatter(millions_formatter)

    # Add data labels to the bars
    for i, v in enumerate(
        sum_charges_by_country['sum(AVERAGE_SUBMITTED_CHRG_AMT)']):
        ax.text(i, v, '${:.2f}M'.format(v * 1e-6),
                ha='center', va='bottom', fontsize=12)

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_top_us_cities_by_submitted_charges(data, n=10):
    """Plots a bar chart of the top N US cities based on total 
    submitted charges.
    """
    # Filter out the United States (US) from the data
    data = (data.select('NPPES_PROVIDER_COUNTRY', 'NPPES_PROVIDER_CITY', 
                        'AVERAGE_SUBMITTED_CHRG_AMT')
            .where(data.NPPES_PROVIDER_COUNTRY == 'US'))

    # Group the data by city and calculate the sum of submitted charges
    sum_charges_by_city = (data.groupby('NPPES_PROVIDER_CITY')
                              .agg({'AVERAGE_SUBMITTED_CHRG_AMT': 'sum'})
                              .sort('sum(AVERAGE_SUBMITTED_CHRG_AMT)', 
                                    ascending=False)
                              .limit(n)
                              .toPandas())

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the bar chart
    ax.bar(sum_charges_by_city['NPPES_PROVIDER_CITY'], 
           sum_charges_by_city['sum(AVERAGE_SUBMITTED_CHRG_AMT)'], 
           color='#66b3ff')

    # Set the labels and title
    ax.set_xlabel('US City', fontsize=16)
    ax.set_ylabel('Total Submitted Charges', fontsize=16)
    ax.set_title('Top {} US Cities by Total Submitted Charges'.format(n), 
                 fontsize=20, fontweight='bold', color='#0d2599')

    # Rotate x-axis labels for better visibility
    ax.set_xticklabels(sum_charges_by_city['NPPES_PROVIDER_CITY'], 
                        rotation=10, 
                        ha='center')

    # Format y-axis labels in millions
    ax.yaxis.set_major_formatter(millions_formatter)

    # Add data labels to the bars
    for i, v in enumerate(
        sum_charges_by_city['sum(AVERAGE_SUBMITTED_CHRG_AMT)']):
        ax.text(i, v, '${:.2f}M'.format(v * 1e-6), ha='center', 
                va='bottom', fontsize=12)

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_top_prescribed_drugs(data, n=10):
    """Plots a horizontal bar chart of the top N most prescribed drugs 
    by total claim count.
    """
    # Group the data by drug name and calculate the total claim count
    claim_count_by_drug = (data.groupby(['generic_name', 'drug_name'])
                           .agg({'total_claim_count': 'sum'})
                           .sort('sum(total_claim_count)', ascending=False)
                           .limit(n)
                           .toPandas())

    # Convert the total claim count to millions
    claim_count_by_drug['sum(total_claim_count)'] = \
    claim_count_by_drug['sum(total_claim_count)'] / 1e6


    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the horizontal bar chart
    ax.barh(claim_count_by_drug['generic_name'], 
            claim_count_by_drug['sum(total_claim_count)'], 
            color=['#66b3ff']*6+['#ff9999',
                                 '#e0eaf8']+['#66b3ff']*1+['#e0eaf8']*1)

    # Set the labels and title
    ax.set_xlabel('Total Claim Count (in millions)', fontsize=16)
    ax.set_ylabel('Drug', fontsize=16)
    ax.set_title(f"Top 10 Drug Claims Reveal Prevalence "
                 f"of Chronic Diseases", fontsize=20, 
                 color='#0d2599', fontweight='bold')

    # Add data labels to the bars
    for i, v in enumerate(claim_count_by_drug['sum(total_claim_count)']):
        ax.text(v, i, '{:.2f}M'.format(v), ha='right', va='center', 
                fontsize=12)

    # Reverse the y-axis
    ax.invert_yaxis()

    # Move the y-axis to the right
    ax.yaxis.tick_right()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_top_specialties(provider_drug):
    """Plots the top 10 specialty descriptions based on total claims for both 
    'Age below 65' and 'Age above 65' in a single plot with shared twin axes.
    """
    # Perform the aggregation and pivot
    top_specialties = (provider_drug.groupby(['specialty_description', 
                                              'ge65_suppress_flag'])
                       .agg(count('*').alias('count'))
                       .withColumn('Age Group', 
                                   when(col('ge65_suppress_flag') == '*', 
                                        'Age below 65')
                                   .otherwise('Age 65 and above'))
                       .drop('ge65_suppress_flag')
                       .groupby('specialty_description')
                       .pivot('Age Group', ['Age below 65', 
                                            'Age 65 and above'])
                       .agg(sum('count').alias('Total Claims'))
                       .withColumn('Total', 
                                   col(
                                       'Age below 65')+col(
                                       'Age 65 and above'))
                       .orderBy('Total', ascending=False)
                       .limit(10)
                       .drop('Total'))

    # Convert to pandas DataFrame
    top_specialties_df = top_specialties.toPandas()

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot the bars for 'Age above 65'
    ax1.bar(top_specialties_df['specialty_description'], 
            top_specialties_df['Age 65 and above'], 
            color='#66b3ff', label='Age 65 and above')
    ax1.set_xlabel('Specialty Description', fontsize=16, color='#0d2599')
    ax1.set_ylabel('Total Claims: Age 65 and above', 
                   fontsize=16, 
                   color='#66b3ff')
    ax1.set_title("Top 10 Specialties' Total Claims by Age Group", 
                  fontsize=20, 
                  fontweight='bold',
                  color='#0d2599')
    ax1.set_xticklabels(top_specialties_df['specialty_description'], 
                        rotation=20, color='#0d2599')  # Set tick label color
        
    # Plot the line for 'Age below 65' with marker 'o'
    ax2.plot(top_specialties_df['specialty_description'], 
             top_specialties_df['Age below 65'], 
             color='#ff0000', label='Age below 65', marker='o')
    ax2.set_ylabel('Total Claims: Age below 65', 
                   fontsize=16, 
                   color='#ff0000')

    # Share y-axis scale between the two subplots
    ax2.sharey(ax1)
    
    # Update colors of y tickmarks
    ax1.tick_params(axis='y', labelcolor='#66b3ff')
    ax2.tick_params(axis='y', labelcolor='#ff0000')
    
    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Format y-axis tick labels in millions
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Adjust the spacing between the bars
    plt.tight_layout()

    # Show the plot
    plt.show()
```
[Back to TOC](#TContents)

<a id='Problem'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">1. PROBLEM STATEMENT</h1>

With an increasingly competitive insurance landscape, insurance companies are continuously seeking ways to optimize their policies and coverage options to remain profitable while providing value to their customers. This involves navigating key challenges, including accurately assessing risk, retaining customers by meeting their unique needs, effectively managing and leveraging vast amounts of data, and innovating product development to align with evolving customer preferences.

To address these challenges, the central question that needs to be answered is:

**How can insurance companies optimize their policies and coverage options based on the analysis of CMS Medicare data?**

By leveraging CMS Medicare data, insurance companies can gain valuable insights into healthcare consumption patterns, risk factors, and customer preferences, thus enabling them to tailor their policies and coverage options effectively, mitigate risks, and ultimately enhance profitability.

[Back to TOC](#TContents)

<a id='Motivation'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">2. MOTIVATION</h1>

In the competitive landscape of the insurance industry and the amount of big data it produces, the use of robust data analytics to optimize policies and coverage options is no longer a luxury — it's a necessity. The potential for improvement is immense and the impact is multi-faceted.

By leveraging CMS Medicare data, the team can tap into a rich source of insights to help insurance companies stay ahead of the curve. These insights can lead to enhanced risk assessment, targeted coverage options, and better customer retention. The result? An edge in a competitive market, improved profitability, and a stronger relationship with customers.

However, this initiative isn't solely about boosting profits. The team is also aiming for a healthier insurance landscape, where companies can swiftly adapt to consumer needs and market changes. By resolving this problem, the team can contribute to the industry's evolution and create a positive, lasting impact on the way insurance companies serve their customers.

This endeavor promises a win-win scenario: insurance companies will thrive in a dynamic market, and customers will benefit from optimized, personalized services. That's a goal worth pursuing for the team.

[Back to TOC](#TContents)

<a id='Methodology'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">3. METHODOLOGY</h1>

<img src='/images/cms/2.png'>
<left><span style="color:gray; font-size:16px"><b>Figure 1. </b>Methodology</span></left>

As illustrated in **Figure 1: Methodology**, this section consists of a three-step process. It begins with data collection, followed by preprocessing of publicly available Centers for Medicare & Medicaid Services (CMS) datasets, and concludes with data analysis involving descriptive analytics and visualization.

Refer to Table 1: Methodology Details for more details about each step.

<left><span style="color:gray; font-size:16px"><b>Table 1. </b>Methodology Details</span></left>

<table>
    <tr style="color:#0d2599; background-color:#e0eaf8">
        <th style="text-align: left">NO.</th>
        <th style="text-align: left">STEP</th>
        <th style="text-align: left">DESCRIPTION</th>
    </tr>
    <tr>
        <td style="text-align: left; font-weight: bold; color:#0d2599">1.</td>
        <td style="text-align: left; font-weight: bold; color:#0d2599">Data Collection</td>
        <td style="text-align: left; ">Collect, assign schemas, and load the CMS datasets into Spark DataFrames. </td>
    </tr>
    <tr>
        <td style="text-align: left; font-weight: bold; color:#0d2599">2.</td>
        <td style="text-align: left; font-weight: bold; color:#0d2599">Data Preprocessing and Exploratory Data Analysis (EDA)</td>
        <td style="text-align: left; ">Clean and preprocess the collected data to prepare it for analysis as well as perform simple EDA which includes handling missing values, correcting inconsistent or incorrect values, and filtering irrelevant records (e.g., remove records with incomplete data).</td>
    </tr>
    <tr>
        <td style="text-align: left; font-weight: bold; color:#0d2599">3.</td>
        <td style="text-align: left; font-weight: bold;  color:#0d2599">Data Analysis & Visualization</td>
        <td style="text-align: left; ">Utilize Apache Spark SQL or Spark DataFrame for descriptive analytics to uncover patterns and trends within the data. Conduct an analysis of procedure distribution, provider types, and costs to enhance insurance policies and coverage options. Finally, employ data visualization techniques to present the findings in a visually informative manner.</td>
    </tr>
</table>

[Back to TOC](#TContents)

<a id='Collection'></a>
<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">4. DATA COLLECTION</h1>

The first step in the methodology is to collect the necessary data. In this case, the team will be focusing on CMS Medicare datasets, specifically the `Medicare Physician & Other Practitioners - by Provider and Service` and `Medicare Part D Prescribers - by Provider and Drug` datasets. These datasets will provide a comprehensive picture of the various policies, coverage options, and prescription drug usage patterns among Medicare beneficiaries.

The **three key steps** involved in data collection section are separated into the following:
- **Identifying the data source**
- **Describing the data**
- **Collecting the data**

<a id='Source'></a>
<h2 style="color:#0d2599; font-size:26px;">4.1 Data Source</h2>

The team utilized the available `Medicare Physician & Other Practitioners - by Provider and Service` (years 2013 to 2015) and `Medicare Part D Prescribers - by Provider and Drug` (years 2013 to 2016) sourced via the jojie-collected public datasets (directory: `/mnt/data/public/cms-gov/`) of the Asian Institute of Management (AIM).

The datasets are in text file format, separated by tab, which were easily collected and processed using Apache Spark.

Specifically, the team chose and explored the following datasets containing Medicare utilization and prescribed drug features:

- **Medicare Physician & Other Practitioners - by Provider and Service [("Medicare Physician & Other Practitioners - by Provider and Service," n.d.)](#Services)** 

    (directory: `/mnt/data/public/cms-gov/Medicare_Provider_Util_Payment_PUF_CY201[3-5]/*.txt`)

    This is a dataset that contains information on the utilization, payments, and submitted charges for procedures and services provided to Medicare beneficiaries by healthcare providers. The data includes information on healthcare providers, such as National Provider Identifier (NPI), provider type, and geographic location for the years 2013 to 2015. Additionally, the data includes information on the services and procedures provided, such as place of service, number of services provided to beneficiaries, and HCPCS (Healthcare Common Procedure Coding System) codes that are used to identify specific medical procedures, services, and supplies.
<!br>
    

- **Medicare Part D Prescribers - by Provider and Drug [("Medicare Part D Prescribers - by Provider and Drug," n.d.)](#Drugs)** 

    (directory: `/mnt/data/public/cms-gov/PartD_Prescriber_PUF_NPI_DRUG_1[3-6]/*.txt`)

    This is a dataset that contains information on the prescription drugs prescribed by healthcare providers who participate in Medicare Part D, which is the prescription drug benefit program for Medicare beneficiaries. Medicare Part D is designed to help individuals afford their prescription drug costs and is provided through private insurance companies that contract with Medicare. This also contains the total number of prescription fills that were dispensed and the total drug cost paid organized by prescribing National Provider Identifier (NPI), drug brand name (if applicable) and drug generic name for the years 2013 to 2016.



<h2>ACKNOWLEDGEMENT</h2>

I completed this project with my Sub Learning Teammate, <b>Erika G. Lacson</b>.

