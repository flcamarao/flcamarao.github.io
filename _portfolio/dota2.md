---
title: "Empowering Dota 2 Match Analysis: Unleashing the Potential of LightGBM Machine Learning Techniques"
excerpt: "Application of machine learning model for accurate winner prediction in competitive gaming <br /><img src='/images/dota2/1.png'>"
collection: portfolio
---

# EXECUTIVE SUMMARY

Dota 2, a leading multiplayer online battle arena (MOBA) game, engages two teams of five players in a strategic contest to destroy their opponent's "Ancient" structure while safeguarding their own. With its prominent presence in the esports domain, insights into the determinants of match outcomes hold significant value.

In each match, Radiant and Dire teams consist of five players who adopt specialized roles based on their chosen heroes. The game map comprises diverse elements, including team bases, lanes, shops, and Roshan's lair. Players focus on hero upgrades, item acquisitions, and enemy base destruction to achieve victory.

This study employs machine learning algorithms to analyze collected data, establish baseline scores, and optimize model performance, yielding high-accuracy models for predicting Dota 2 match outcomes. Nevertheless, the reduction of the dataset due to time constraints and computational resources presents a limitation to the study's scope.

Future research can investigate additional models and optimization approaches to augment or supplement existing findings, further advancing our understanding and predictive capabilities for Dota 2 match outcomes.

<img src='/images/dota2/2.png'>


# HIGHLIGTHS

1. Showcased the application of Bayesian Optimization and its impact on enhancing accuracy levels.
2. Illustrated the implementation of Light Gradient Boosting alongside hyperparameter tuning.
3. Explored the utilization of various Classification Models for diverse predictions.
4. Identified gold as the most influential predictor in determining match outcomes.
5. Emphasized the importance of Feature Engineering in optimizing model performance and efficiency.

# METHODOLOGY

The overarching methodology of this project focuses on employing various machine learning models, particularly LightGBM, to accurately predict Dota 2 match outcomes. The following steps outline the process:

- Data Retrieval: Acquiring the relevant dataset for analysis.
- Data Cleaning: Ensuring data quality by removing inconsistencies and inaccuracies.
- Exploratory Data Analysis: Investigating data patterns, relationships, and trends to gain insights.
- Data Preprocessing: Transforming and preparing the data for machine learning models.
- ML Models Simulation: Implementing and comparing the performance of various machine learning models.
- Hyperparameter Optimization: Fine-tuning model parameters to enhance predictive accuracy and efficiency.

## Import Libraries
```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning models and algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier,
                              VotingClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Machine learning tools and utilities
from mltools import *
from bayes_opt import BayesianOptimization

# Data preprocessing and preparation
from sklearn.preprocessing import StandardScaler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

## Data Retrieval

The data used in this project is sourced from a Kaggle dataset [5]. The OpenDota API, which could have been an alternative, is no longer a viable option due to its restrictions on non-premium subscribers and the removal of free access to the API.

### Data Loading
```python
# loading data
train_data = pd.read_csv('./train_features.csv', index_col='match_id_hash')
test_data = pd.read_csv('./test_features.csv', index_col='match_id_hash')
train_y = pd.read_csv('./train_targets.csv', index_col='match_id_hash')

colors = ['#d9534f', '#5cb85c']
# Compute the target value counts
target_counts = train_y['radiant_win'].value_counts()

# Create a plot using seaborn
plt.figure(figsize=(6, 4))
sns.set_style('whitegrid')
ax = sns.barplot(x=target_counts.index,
                 y=target_counts.values, palette=colors)
ax.set_title('Win/Lose Distribution')
ax.set_xlabel('Target')
ax.set_ylabel('Count')
ax.set_xticklabels(['Lose', 'Win'])

# Save the plot as a PNG file
plt.savefig("images/balance.png")

# Create an HTML img tag to display the image
img_tag = (f'<img src="images/balance.png" alt="balance"'
           f'style="display:block; margin-left:auto;'
           f'margin-right:auto; width:80%;">')

# display the HTML <img> tag
display(HTML(img_tag))
plt.close()
```
<img src='/images/dota2/3.png'>

```python
sns.pairplot(train_data.iloc[:,:4], hue='lobby_type')
```
<img src='/images/dota2/4.png'>

### Data Description
```python
num_rows = train_data.shape[0]
num_cols = train_data.shape[1]
html_table = train_data.head().to_html()
html_table_with_info = f"{html_table} \n <p>Number of Rows: {num_rows}<br>Number of Columns: {num_cols}</p>"

# Print the HTML table
print(html_table_with_info)
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_time</th>
      <th>game_mode</th>
      <th>lobby_type</th>
      <th>objectives_len</th>
      <th>chat_len</th>
      <th>r1_hero_id</th>
      <th>r1_kills</th>
      <th>r1_deaths</th>
      <th>r1_assists</th>
      <th>r1_denies</th>
      <th>r1_gold</th>
      <th>r1_lh</th>
      <th>r1_xp</th>
      <th>r1_health</th>
      <th>r1_max_health</th>
      <th>r1_max_mana</th>
      <th>r1_level</th>
      <th>r1_x</th>
      <th>r1_y</th>
      <th>r1_stuns</th>
      <th>r1_creeps_stacked</th>
      <th>r1_camps_stacked</th>
      <th>r1_rune_pickups</th>
      <th>r1_firstblood_claimed</th>
      <th>r1_teamfight_participation</th>
      <th>r1_towers_killed</th>
      <th>r1_roshans_killed</th>
      <th>r1_obs_placed</th>
      <th>r1_sen_placed</th>
      <th>r2_hero_id</th>
      <th>r2_kills</th>
      <th>r2_deaths</th>
      <th>r2_assists</th>
      <th>r2_denies</th>
      <th>r2_gold</th>
      <th>r2_lh</th>
      <th>r2_xp</th>
      <th>r2_health</th>
      <th>r2_max_health</th>
      <th>r2_max_mana</th>
      <th>r2_level</th>
      <th>r2_x</th>
      <th>r2_y</th>
      <th>r2_stuns</th>
      <th>r2_creeps_stacked</th>
      <th>r2_camps_stacked</th>
      <th>r2_rune_pickups</th>
      <th>r2_firstblood_claimed</th>
      <th>r2_teamfight_participation</th>
      <th>r2_towers_killed</th>
      <th>r2_roshans_killed</th>
      <th>r2_obs_placed</th>
      <th>r2_sen_placed</th>
      <th>r3_hero_id</th>
      <th>r3_kills</th>
      <th>r3_deaths</th>
      <th>r3_assists</th>
      <th>r3_denies</th>
      <th>r3_gold</th>
      <th>r3_lh</th>
      <th>r3_xp</th>
      <th>r3_health</th>
      <th>r3_max_health</th>
      <th>r3_max_mana</th>
      <th>r3_level</th>
      <th>r3_x</th>
      <th>r3_y</th>
      <th>r3_stuns</th>
      <th>r3_creeps_stacked</th>
      <th>r3_camps_stacked</th>
      <th>r3_rune_pickups</th>
      <th>r3_firstblood_claimed</th>
      <th>r3_teamfight_participation</th>
      <th>r3_towers_killed</th>
      <th>r3_roshans_killed</th>
      <th>r3_obs_placed</th>
      <th>r3_sen_placed</th>
      <th>r4_hero_id</th>
      <th>r4_kills</th>
      <th>r4_deaths</th>
      <th>r4_assists</th>
      <th>r4_denies</th>
      <th>r4_gold</th>
      <th>r4_lh</th>
      <th>r4_xp</th>
      <th>r4_health</th>
      <th>r4_max_health</th>
      <th>r4_max_mana</th>
      <th>r4_level</th>
      <th>r4_x</th>
      <th>r4_y</th>
      <th>r4_stuns</th>
      <th>r4_creeps_stacked</th>
      <th>r4_camps_stacked</th>
      <th>r4_rune_pickups</th>
      <th>r4_firstblood_claimed</th>
      <th>r4_teamfight_participation</th>
      <th>r4_towers_killed</th>
      <th>r4_roshans_killed</th>
      <th>r4_obs_placed</th>
      <th>r4_sen_placed</th>
      <th>r5_hero_id</th>
      <th>r5_kills</th>
      <th>r5_deaths</th>
      <th>r5_assists</th>
      <th>r5_denies</th>
      <th>r5_gold</th>
      <th>r5_lh</th>
      <th>r5_xp</th>
      <th>r5_health</th>
      <th>r5_max_health</th>
      <th>r5_max_mana</th>
      <th>r5_level</th>
      <th>r5_x</th>
      <th>r5_y</th>
      <th>r5_stuns</th>
      <th>r5_creeps_stacked</th>
      <th>r5_camps_stacked</th>
      <th>r5_rune_pickups</th>
      <th>r5_firstblood_claimed</th>
      <th>r5_teamfight_participation</th>
      <th>r5_towers_killed</th>
      <th>r5_roshans_killed</th>
      <th>r5_obs_placed</th>
      <th>r5_sen_placed</th>
      <th>d1_hero_id</th>
      <th>d1_kills</th>
      <th>d1_deaths</th>
      <th>d1_assists</th>
      <th>d1_denies</th>
      <th>d1_gold</th>
      <th>d1_lh</th>
      <th>d1_xp</th>
      <th>d1_health</th>
      <th>d1_max_health</th>
      <th>d1_max_mana</th>
      <th>d1_level</th>
      <th>d1_x</th>
      <th>d1_y</th>
      <th>d1_stuns</th>
      <th>d1_creeps_stacked</th>
      <th>d1_camps_stacked</th>
      <th>d1_rune_pickups</th>
      <th>d1_firstblood_claimed</th>
      <th>d1_teamfight_participation</th>
      <th>d1_towers_killed</th>
      <th>d1_roshans_killed</th>
      <th>d1_obs_placed</th>
      <th>d1_sen_placed</th>
      <th>d2_hero_id</th>
      <th>d2_kills</th>
      <th>d2_deaths</th>
      <th>d2_assists</th>
      <th>d2_denies</th>
      <th>d2_gold</th>
      <th>d2_lh</th>
      <th>d2_xp</th>
      <th>d2_health</th>
      <th>d2_max_health</th>
      <th>d2_max_mana</th>
      <th>d2_level</th>
      <th>d2_x</th>
      <th>d2_y</th>
      <th>d2_stuns</th>
      <th>d2_creeps_stacked</th>
      <th>d2_camps_stacked</th>
      <th>d2_rune_pickups</th>
      <th>d2_firstblood_claimed</th>
      <th>d2_teamfight_participation</th>
      <th>d2_towers_killed</th>
      <th>d2_roshans_killed</th>
      <th>d2_obs_placed</th>
      <th>d2_sen_placed</th>
      <th>d3_hero_id</th>
      <th>d3_kills</th>
      <th>d3_deaths</th>
      <th>d3_assists</th>
      <th>d3_denies</th>
      <th>d3_gold</th>
      <th>d3_lh</th>
      <th>d3_xp</th>
      <th>d3_health</th>
      <th>d3_max_health</th>
      <th>d3_max_mana</th>
      <th>d3_level</th>
      <th>d3_x</th>
      <th>d3_y</th>
      <th>d3_stuns</th>
      <th>d3_creeps_stacked</th>
      <th>d3_camps_stacked</th>
      <th>d3_rune_pickups</th>
      <th>d3_firstblood_claimed</th>
      <th>d3_teamfight_participation</th>
      <th>d3_towers_killed</th>
      <th>d3_roshans_killed</th>
      <th>d3_obs_placed</th>
      <th>d3_sen_placed</th>
      <th>d4_hero_id</th>
      <th>d4_kills</th>
      <th>d4_deaths</th>
      <th>d4_assists</th>
      <th>d4_denies</th>
      <th>d4_gold</th>
      <th>d4_lh</th>
      <th>d4_xp</th>
      <th>d4_health</th>
      <th>d4_max_health</th>
      <th>d4_max_mana</th>
      <th>d4_level</th>
      <th>d4_x</th>
      <th>d4_y</th>
      <th>d4_stuns</th>
      <th>d4_creeps_stacked</th>
      <th>d4_camps_stacked</th>
      <th>d4_rune_pickups</th>
      <th>d4_firstblood_claimed</th>
      <th>d4_teamfight_participation</th>
      <th>d4_towers_killed</th>
      <th>d4_roshans_killed</th>
      <th>d4_obs_placed</th>
      <th>d4_sen_placed</th>
      <th>d5_hero_id</th>
      <th>d5_kills</th>
      <th>d5_deaths</th>
      <th>d5_assists</th>
      <th>d5_denies</th>
      <th>d5_gold</th>
      <th>d5_lh</th>
      <th>d5_xp</th>
      <th>d5_health</th>
      <th>d5_max_health</th>
      <th>d5_max_mana</th>
      <th>d5_level</th>
      <th>d5_x</th>
      <th>d5_y</th>
      <th>d5_stuns</th>
      <th>d5_creeps_stacked</th>
      <th>d5_camps_stacked</th>
      <th>d5_rune_pickups</th>
      <th>d5_firstblood_claimed</th>
      <th>d5_teamfight_participation</th>
      <th>d5_towers_killed</th>
      <th>d5_roshans_killed</th>
      <th>d5_obs_placed</th>
      <th>d5_sen_placed</th>
    </tr>
    <tr>
      <th>match_id_hash</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a400b8f29dece5f4d266f49f1ae2e98a</th>
      <td>155</td>
      <td>22</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>543</td>
      <td>7</td>
      <td>533</td>
      <td>358</td>
      <td>600</td>
      <td>350.93784</td>
      <td>2</td>
      <td>116</td>
      <td>122</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>399</td>
      <td>4</td>
      <td>478</td>
      <td>636</td>
      <td>720</td>
      <td>254.93774</td>
      <td>2</td>
      <td>124</td>
      <td>126</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>304</td>
      <td>0</td>
      <td>130</td>
      <td>700</td>
      <td>700</td>
      <td>242.93773</td>
      <td>1</td>
      <td>70</td>
      <td>156</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>389</td>
      <td>4</td>
      <td>506</td>
      <td>399</td>
      <td>700</td>
      <td>326.93780</td>
      <td>2</td>
      <td>170</td>
      <td>86</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>77</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>402</td>
      <td>10</td>
      <td>344</td>
      <td>422</td>
      <td>800</td>
      <td>314.93780</td>
      <td>2</td>
      <td>120</td>
      <td>100</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
      <td>982</td>
      <td>12</td>
      <td>780</td>
      <td>650</td>
      <td>720</td>
      <td>386.93787</td>
      <td>3</td>
      <td>82</td>
      <td>170</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>788</td>
      <td>9</td>
      <td>706</td>
      <td>640</td>
      <td>640</td>
      <td>422.93790</td>
      <td>3</td>
      <td>174</td>
      <td>90</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>531</td>
      <td>0</td>
      <td>307</td>
      <td>720</td>
      <td>720</td>
      <td>242.93773</td>
      <td>2</td>
      <td>180</td>
      <td>84</td>
      <td>0.299948</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>84</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>796</td>
      <td>0</td>
      <td>421</td>
      <td>760</td>
      <td>760</td>
      <td>326.93780</td>
      <td>2</td>
      <td>90</td>
      <td>150</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>851</td>
      <td>11</td>
      <td>870</td>
      <td>593</td>
      <td>680</td>
      <td>566.93805</td>
      <td>3</td>
      <td>128</td>
      <td>128</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b9c57c450ce74a2af79c9ce96fac144d</th>
      <td>658</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>15</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>5257</td>
      <td>52</td>
      <td>3937</td>
      <td>1160</td>
      <td>1160</td>
      <td>566.93805</td>
      <td>8</td>
      <td>76</td>
      <td>78</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.437500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3394</td>
      <td>19</td>
      <td>3897</td>
      <td>1352</td>
      <td>1380</td>
      <td>386.93787</td>
      <td>8</td>
      <td>78</td>
      <td>166</td>
      <td>8.397949</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.312500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2212</td>
      <td>4</td>
      <td>2561</td>
      <td>710</td>
      <td>860</td>
      <td>530.93800</td>
      <td>6</td>
      <td>156</td>
      <td>146</td>
      <td>11.964951</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.312500</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>63</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>12</td>
      <td>4206</td>
      <td>38</td>
      <td>4459</td>
      <td>420</td>
      <td>880</td>
      <td>482.93796</td>
      <td>9</td>
      <td>154</td>
      <td>148</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.437500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>89</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>3103</td>
      <td>14</td>
      <td>2712</td>
      <td>856</td>
      <td>900</td>
      <td>446.93793</td>
      <td>6</td>
      <td>150</td>
      <td>148</td>
      <td>21.697395</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.375000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2823</td>
      <td>24</td>
      <td>3281</td>
      <td>700</td>
      <td>700</td>
      <td>686.93820</td>
      <td>7</td>
      <td>88</td>
      <td>170</td>
      <td>3.165901</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>2466</td>
      <td>17</td>
      <td>2360</td>
      <td>758</td>
      <td>1040</td>
      <td>326.93780</td>
      <td>6</td>
      <td>156</td>
      <td>98</td>
      <td>0.066650</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.25</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>3624</td>
      <td>29</td>
      <td>3418</td>
      <td>485</td>
      <td>800</td>
      <td>350.93784</td>
      <td>7</td>
      <td>124</td>
      <td>144</td>
      <td>0.299955</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>56</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2808</td>
      <td>18</td>
      <td>2730</td>
      <td>567</td>
      <td>1160</td>
      <td>410.93790</td>
      <td>6</td>
      <td>124</td>
      <td>142</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>92</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1423</td>
      <td>8</td>
      <td>1136</td>
      <td>800</td>
      <td>800</td>
      <td>446.93793</td>
      <td>4</td>
      <td>180</td>
      <td>176</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6db558535151ea18ca70a6892197db41</th>
      <td>21</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>101</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>680</td>
      <td>680</td>
      <td>506.93800</td>
      <td>1</td>
      <td>118</td>
      <td>118</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>720</td>
      <td>720</td>
      <td>278.93777</td>
      <td>1</td>
      <td>156</td>
      <td>104</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>568</td>
      <td>600</td>
      <td>254.93774</td>
      <td>1</td>
      <td>78</td>
      <td>144</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>580</td>
      <td>580</td>
      <td>254.93774</td>
      <td>1</td>
      <td>150</td>
      <td>78</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>580</td>
      <td>580</td>
      <td>374.93787</td>
      <td>1</td>
      <td>78</td>
      <td>142</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>660</td>
      <td>660</td>
      <td>266.93774</td>
      <td>1</td>
      <td>180</td>
      <td>178</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>586</td>
      <td>620</td>
      <td>278.93777</td>
      <td>1</td>
      <td>100</td>
      <td>174</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>660</td>
      <td>660</td>
      <td>290.93777</td>
      <td>1</td>
      <td>178</td>
      <td>112</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>600</td>
      <td>600</td>
      <td>302.93777</td>
      <td>1</td>
      <td>176</td>
      <td>110</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>640</td>
      <td>640</td>
      <td>446.93793</td>
      <td>1</td>
      <td>162</td>
      <td>162</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46a0ddce8f7ed2a8d9bd5edcbb925682</th>
      <td>576</td>
      <td>22</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1613</td>
      <td>0</td>
      <td>1471</td>
      <td>900</td>
      <td>900</td>
      <td>290.93777</td>
      <td>4</td>
      <td>170</td>
      <td>96</td>
      <td>2.366089</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2816</td>
      <td>30</td>
      <td>3602</td>
      <td>878</td>
      <td>1100</td>
      <td>494.93796</td>
      <td>8</td>
      <td>82</td>
      <td>154</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.285714</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>101</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>4017</td>
      <td>44</td>
      <td>4811</td>
      <td>980</td>
      <td>980</td>
      <td>902.93835</td>
      <td>9</td>
      <td>126</td>
      <td>128</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1558</td>
      <td>2</td>
      <td>1228</td>
      <td>640</td>
      <td>640</td>
      <td>422.93790</td>
      <td>4</td>
      <td>120</td>
      <td>138</td>
      <td>7.098264</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>3344</td>
      <td>55</td>
      <td>3551</td>
      <td>1079</td>
      <td>1100</td>
      <td>362.93784</td>
      <td>7</td>
      <td>176</td>
      <td>94</td>
      <td>1.932884</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.142857</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2712</td>
      <td>69</td>
      <td>2503</td>
      <td>825</td>
      <td>1160</td>
      <td>338.93784</td>
      <td>6</td>
      <td>94</td>
      <td>158</td>
      <td>0.000000</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>98</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2217</td>
      <td>23</td>
      <td>3310</td>
      <td>735</td>
      <td>880</td>
      <td>506.93800</td>
      <td>7</td>
      <td>126</td>
      <td>142</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>3035</td>
      <td>44</td>
      <td>2508</td>
      <td>817</td>
      <td>860</td>
      <td>350.93784</td>
      <td>6</td>
      <td>78</td>
      <td>160</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2004</td>
      <td>16</td>
      <td>1644</td>
      <td>1160</td>
      <td>1160</td>
      <td>386.93787</td>
      <td>4</td>
      <td>176</td>
      <td>100</td>
      <td>4.998863</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1333</td>
      <td>2</td>
      <td>1878</td>
      <td>630</td>
      <td>740</td>
      <td>518.93800</td>
      <td>5</td>
      <td>82</td>
      <td>160</td>
      <td>8.664527</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b1b35ff97723d9b7ade1c9c3cf48f770</th>
      <td>453</td>
      <td>22</td>
      <td>7</td>
      <td>1</td>
      <td>3</td>
      <td>42</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1404</td>
      <td>9</td>
      <td>1351</td>
      <td>1000</td>
      <td>1000</td>
      <td>338.93784</td>
      <td>4</td>
      <td>80</td>
      <td>164</td>
      <td>9.930903</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1840</td>
      <td>14</td>
      <td>1693</td>
      <td>868</td>
      <td>1000</td>
      <td>350.93784</td>
      <td>5</td>
      <td>78</td>
      <td>166</td>
      <td>1.832892</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1204</td>
      <td>10</td>
      <td>3210</td>
      <td>578</td>
      <td>860</td>
      <td>792.93823</td>
      <td>7</td>
      <td>120</td>
      <td>122</td>
      <td>3.499146</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1724</td>
      <td>21</td>
      <td>1964</td>
      <td>777</td>
      <td>980</td>
      <td>434.93793</td>
      <td>5</td>
      <td>138</td>
      <td>94</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1907</td>
      <td>8</td>
      <td>1544</td>
      <td>281</td>
      <td>820</td>
      <td>446.93793</td>
      <td>4</td>
      <td>174</td>
      <td>100</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1422</td>
      <td>10</td>
      <td>1933</td>
      <td>709</td>
      <td>940</td>
      <td>362.93784</td>
      <td>5</td>
      <td>84</td>
      <td>170</td>
      <td>11.030720</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1457</td>
      <td>12</td>
      <td>1759</td>
      <td>712</td>
      <td>820</td>
      <td>482.93796</td>
      <td>5</td>
      <td>174</td>
      <td>106</td>
      <td>2.199399</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2402</td>
      <td>35</td>
      <td>3544</td>
      <td>349</td>
      <td>720</td>
      <td>434.93793</td>
      <td>7</td>
      <td>128</td>
      <td>126</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1697</td>
      <td>12</td>
      <td>1651</td>
      <td>680</td>
      <td>680</td>
      <td>374.93787</td>
      <td>4</td>
      <td>176</td>
      <td>108</td>
      <td>13.596678</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>2199</td>
      <td>32</td>
      <td>1919</td>
      <td>692</td>
      <td>740</td>
      <td>302.93777</td>
      <td>5</td>
      <td>104</td>
      <td>162</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table> 
 <p>Number of Rows: 39675<br>Number of Columns: 245</p>
 </div>

 The dataset comprises 39,675 rows and 245 columns or features, representing various aspects of Dota 2 matches. Each match involves 10 players, distributed across two teams of five players each. For every player, there are 24 unique features, leading to a total of 240 columns representing player-specific data.

A detailed description of each player-specific feature can be found in the provided reference table [6].

|  Feature  | Description |
| ------------- |:-------------| 
| **hero_id** | ID of player's hero (int64). [Heroes](https://dota2.gamepedia.com/Heroes) are the essential element of Dota 2, as the course of the match is dependent on their intervention. During a match, two opposing teams select five out of 117 heroes that accumulate experience and gold to grow stronger and gain new abilities in order to destroy the opponent's Ancient. Most heroes have a distinct role that defines how they affect the battlefield, though many heroes can perform multiple roles. A hero's appearance can be modified with equipment.|
| **kills** | Number of killed players (int64).|
| **deaths** | Number of deaths of the player (int64).|
| **gold** | Amount of gold (int64). [Gold](https://dota2.gamepedia.com/Gold) is the currency used to buy items or instantly revive your hero. Gold can be earned from killing heroes, creeps, or buildings. |
| **xp** | Experience points (int64). [Experience](https://dota2.gamepedia.com/Experience) is an element heroes can gather by killing enemy units, or being present as enemy units get killed. On its own, experience does nothing, but when accumulated, it increases the hero's level, so that they grow more powerful.   |
| **lh** | Number of last hits (int64). [Last-hitting](https://dota2.gamepedia.com/Creep_control_techniques#Last-hitting) is a technique where you (or a creep under your control) get the 'last hit' on a neutral creep, enemy lane creep, or enemy hero. The hero that dealt the killing blow to the enemy unit will be awarded a bounty.|
| **denies** | Number of denies (int64). [Denying](https://dota2.gamepedia.com/Denying) is the act of preventing enemy heroes from getting the last hit on a friendly unit by last hitting the unit oneself. Enemies earn reduced experience if the denied unit is not controlled by a player, and no experience if it is a player controlled unit. Enemies gain no gold from any denied unit. |
| **assists** | Number of [assists](https://dota2.gamepedia.com/Gold#Assists_.28AoE_gold.29) (int64). Allied heroes within 1300 radius of a killed enemy, including the killer, receive experience and reliable gold if they assisted in the kill. To qualify for an assist, the allied hero merely has to be within the given radius of the dying enemy hero. |
| **health** | Health points (int64). [Health](https://dota2.gamepedia.com/Health) represents the life force of a unit. When a unit's current health reaches 0, it dies. Every hero has a base health pool of 200. This value exists for all heroes and cannot be altered. This means that a hero's maximum health cannot drop below 200. |
| **max_health** | Hero's maximum health pool (int64).|
| **max_mana** | Hero's maximum mana pool (float64). [Mana](https://dota2.gamepedia.com/Mana) represents the magic power of a unit. It is used as a cost for the majority of active and even some passive abilities. Every hero has a base mana pool of 75, while most non-hero units only have a set mana pool if they have abilities which require mana, with a few exceptions. These values cannot be altered. This means that a hero's maximum mana cannot drop below 75. |
| **level** | [Level](https://dota2.gamepedia.com/Experience#Leveling) of player's hero (int64). Each hero begins at level 1, with one free ability point to spend. Heroes may level up by acquiring certain amounts of experience. Upon leveling up, the hero's attributes increase by fixed amounts (unique for each hero), which makes them overall more powerful. Heroes may also gain more ability points by leveling up, allowing them to learn new spells, or to improve an already learned spell, making it more powerful. Heroes can gain a total for 24 levels, resulting in level 25 as the highest possible level a hero can reach. |
| **x** | Player's X coordinate (int64) |
| **y** | Player's Y coordinate (int64) |
| **stuns** | Total stun duration of all stuns (float64). [Stun](https://dota2.gamepedia.com/Stun) is a status effect that completely locks down affected units, disabling almost all of its capabilities. |
| **creeps_stacked** | Number of stacked creeps (int64). [Creep Stacking](https://dota2.gamepedia.com/Creep_Stacking) is the process of drawing neutral creeps away from their camps in order to increase the number of units in an area. By pulling neutral creeps beyond their camp boundaries, the game will generate a new set of creeps for the player to interact with in addition to any remaining creeps. This is incredibly time efficient, since it effectively increases the amount of gold available for a team. |
| **camps_stacked** | Number of stacked camps  (int64). |
| **rune_pickups** | Number of picked up [runes](https://dota2.gamepedia.com/Runes)  (int64).  |
| **firstblood_claimed** | boolean feature? (int64) |
| **teamfight_participation** |  Team fight participation rate? (float64) |
| **towers_killed** | Number of killed/destroyed Towers (int64). [Towers](https://dota2.gamepedia.com/Buildings#Towers) are the main line of defense for both teams, attacking any non-neutral enemy that gets within their range. Both factions have all three lanes guarded by three towers each. Additionally, each faction's Ancient have two towers as well, resulting in a total of 11 towers per faction. Towers come in 4 different tiers. |
| **roshans_killed** | Number of killed Roshans  (int64). [Roshan](https://dota2.gamepedia.com/Roshan) is the most powerful neutral creep in Dota 2. It is the first unit which spawns, right as the match is loaded. During the early to mid game, he easily outmatches almost every hero in one-on-one combat. Very few heroes can take him on alone during the mid-game. Even in the late game, lots of heroes struggle fighting him one on one, since Roshan grows stronger as time passes. |
| **obs_placed** | Number of observer-wards placed by a player (int64). [Observer Ward](https://dota2.gamepedia.com/Observer_Ward), an invisible watcher that gives ground vision in a 1600 radius to your team. Lasts 6 minutes. |
| **sen_placed** | Number of sentry-wards placed by a player (int64) [Sentry Ward](https://dota2.gamepedia.com/Sentry_Ward), an invisible watcher that grants True Sight, the ability to see invisible enemy units and wards, to any existing allied vision within a radius. Lasts 6 minutes.|

## Data Cleaning
```python
train_data.info()
train_data.describe()
train_data.isnull().sum()
train_data.drop_duplicates()
```

## Exploratory Data Analysis
```python
# Create a countplot with a custom color palette
sns.countplot(data=train_data, x='lobby_type', order=train_data['lobby_type'].value_counts().index, palette='Set2');

# Add a title to the plot
plt.title('Counts of games in lobby type');

# Show the plot
plt.show()
```
<img src='/images/dota2/5.png'>

```python
# Create a countplot with the 'Set2' color palette for game modes
sns.countplot(data=train_data, x='game_mode',
              order=train_data['game_mode'].value_counts().index, palette='Set2')

# Add a title to the plot
plt.title('Counts of games in different modes')

# Show the plot
plt.show()

# Filter the dataset to only include the most common game mode
most_common_game_mode = train_data['game_mode'].value_counts().idxmax()
filtered_train_data = train_data[train_data['game_mode'] == most_common_game_mode]
```
<img src='/images/dota2/6.png'>

**Insight**
- Different lobby types and game modes have distinct criteria; therefore, selecting the lobby type and game mode with the highest count ensures a balanced dataset with more consistent datapoints. This approach minimizes potential biases and variations that could arise from considering multiple lobby types and game modes, leading to more reliable predictions and analysis.

## Data Preprocessing

### Feature Transformation
```python
# remove lobby_type with lower counts
train_y['lobby_type'] = train_data['lobby_type']
train_data = train_data[train_data['lobby_type'] == 7]
test_data = test_data[test_data['lobby_type'] == 7]
train_y = train_y[train_y['lobby_type'] == 7]

# remove game_mode with lower counts
train_y['game_mode'] = train_data['game_mode']
train_data = train_data[train_data['game_mode'] == 22]
test_data = test_data[test_data['game_mode'] == 22]
train_y = train_y[train_y['game_mode'] == 22]
# drop lobby_type and game_mode for train_y
train_y = train_y.drop(columns=['lobby_type', 'game_mode'])
# mapping win and lose values
train_y = train_y['radiant_win'].map({True: 1, False:0})

# Get the unique values of 'radiant_win' column
unique_vals = train_y.reset_index()['radiant_win'].unique()

# Get the count of win for each unique value of 'radiant_win'
win_counts = train_y.reset_index()['radiant_win'].value_counts()

# Create a bar plot with the 'Set2' color palette for Dire and Radiant wins
sns.barplot(x=unique_vals, y=win_counts, palette='Set2')

# Set the x-axis tick labels to 'Dire Win' and 'Radiant Win'
plt.xticks([0, 1], labels=['Dire Win', 'Radiant Win'])

# Add labels to the x- and y-axes and a title to the plot
plt.xlabel('Dire & Radiant')
plt.ylabel('Win Counts')
plt.title('Dire vs Radiant Win Counts')

# Show the plot
plt.show()
```

<img src='/images/dota2/7.png'>

### Feature Engineering
```python
# combing all individual character features into a team based features
feature_names = train_data.columns
num = []
for y in range(24):
    for i in [feature_names[feature_names.str.contains("r"+str(i)) == True] for i in range(1,6)]:
        num.append(i[y])
    col = num[0].split("_")[1]
    train_data['r_'+col] = train_data[num].sum(axis=1)
    test_data['r_'+col] = test_data[num].sum(axis=1)
    
    # dropping individual features
    train_data.drop(columns=num, inplace=True)
    test_data.drop(columns=num, inplace=True)
    num = []
    
for y in range(24):
    for i in [feature_names[feature_names.str.contains("d"+str(i)) == True] for i in range(1,6)]:
        num.append(i[y])
    col = num[0].split("_")[1]
    train_data['d_'+col] = train_data[num].sum(axis=1)
    test_data['d_'+col] = test_data[num].sum(axis=1)
    
    # dropping individual features
    train_data.drop(columns=num, inplace=True)
    test_data.drop(columns=num, inplace=True)
    num = []
```

### Feature Selection
```python
# what features to be used
to_load = (['r_kills', 'r_deaths', 'r_assists', 'r_denies', 'r_lh', 'r_gold',
            'd_kills', 'd_deaths', 'd_assists', 'd_denies', 'd_lh', 'd_gold'])
train_X = train_data[to_load]
test_X = test_data[to_load]

# reduced the datapoints for the interest of runtime and
# to show the significance of the models
feature_names = train_X.columns
train_sample = int(train_X.shape[0]/2)
test_sample = int(test_X.shape[0]/2)

# getting sample size
train_X = train_X.sample(train_sample, random_state=3)
train_y = train_y.sample(train_sample, random_state=3)
test_X = test_X.sample(test_sample, random_state=3)
train_X
```

### Data Scaling
```python
# Scaling
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.fit_transform(test_X)
```
<img src='/images/dota2/8.png'>
    
## Final Data
```python
# scaled data
train_X_scaled # for train value
test_X_scaled # for test value

# original data
train_X # for train value
test_X # for test value

# output to csv
train_X.to_csv('train_X.csv')
test_X.to_csv('test_X.csv')
train_y.to_csv('train_y.csv')
```

# RESULTS AND DISCUSSION

## Auto ML Simulation
```python
# Select methods
methods = ['kNN', 'Logistic (L1)', 'Logistic (L2)', 'Decision Tree',
           'RF Classifier', 'GB Classifier', 'XGB Classifier',
           'AdaBoost DT', 'LightGBM Classifier', 'CatBoost Classifier']

# Perform training and testing
ml_models = MLModels.run_classifier(
    train_X_scaled, train_y, feature_names, task='C',
    use_methods=methods, n_trials=2, tree_rs=3, test_size=0.20,
    n_neighbors=list(range(1, 3)),
    C=[1e-1, 1],
    max_depth=[5, 10])
res = MLModels.summarize(ml_models, feature_names,
                         show_plot=True, show_top=True)
```
<img src='/images/dota2/9.png'>
<img src='/images/dota2/10.png'>
<img src='/images/dota2/11.png'>
<img src='/images/dota2/12.png'>
<img src='/images/dota2/13.png'>
<img src='/images/dota2/14.png'>
<img src='/images/dota2/15.png'>
<img src='/images/dota2/16.png'>
<img src='/images/dota2/17.png'>
<img src='/images/dota2/18.png'>
<img src='/images/dota2/19.png'>

## Feature Importance
```python
for model_name in methods[3:]:
    try:
        ax = ml_models[model_name].plot_feature_importance(feature_names)
        ax.set_title(model_name, fontsize=16, weight="bold")
    except Exception as e:
        print(model_name, e)
```
<img src='/images/dota2/20.png'>
<img src='/images/dota2/21.png'>
<img src='/images/dota2/22.png'>
<img src='/images/dota2/23.png'>
<img src='/images/dota2/24.png'>
<img src='/images/dota2/25.png'>
<img src='/images/dota2/26.png'>

## Light GBM Classifier Simulation
```python
# reduced params range based from previous multiple trials
tune_model(train_X_scaled, train_y, 'Classification', 'LightGBM Classifier',
           params={'max_depth': [25, 50],
                   'n_estimators': [150, 200],
                   'learning_rate': [0.1, 0.2]},
           n_trials=2, tree_rs=3)
```
<div>
<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Best Parameter</th>
  </tr>
  <tr>
    <td>LightGBM Classifier</td>
    <td>69.41%</td>
    <td>{'max_depth': 25, 'n_estimators': 200, 'learning_rate': 0.1}</td>
  </tr>
  <tr>
    <td>Classifier</td>
    <td colspan="2">LGBMClassifier(max_depth=25, n_estimators=200, random_state=3)</td>
  </tr>
  <tr>
    <td>acc</td>
    <td>69.41</td>
    <td></td>
  </tr>
  <tr>
    <td>std</td>
    <td>0.0013953488372092648</td>
    <td></td>
  </tr>
</table>
</div>

## Bayesian Optimization Simulation
In this project, we demonstrated the use of Bayesian optimization as a function optimization package by optimizing three hyperparameters of the LightGBM classifier. This was done to show how Bayesian optimization can improve the accuracy of a machine learning model by finding the optimal combination of hyperparameters.

### The function to be optimized
```python
def lgb_cv(num_leaves, max_depth, min_data_in_leaf):
    params = {
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "learning_rate": 0.5,
        'min_data_in_leaf': int(min_data_in_leaf),
        "force_col_wise": True,
        'verbose': -1,
        "metric" : "auc",
        "objective" : "binary",
    }
    
    lgtrain = lightgbm.Dataset(train_X_scaled, train_y)
    cv_result = lightgbm.cv(params,
                       lgtrain,
                       200,
                       early_stopping_rounds=200,
                       stratified=True,
                       nfold=5)
    return cv_result['auc-mean'][-1]
```
### The optimizer function
```python
def bayesian_optimizer(init_points, num_iter, **args):
    lgb_BO = BayesianOptimization(lgb_cv, {'num_leaves': (100, 200),
                                           'max_depth': (25, 50),
                                           'min_data_in_leaf': (50, 200)
                                           })
    lgb_BO.maximize(init_points=init_points, n_iter=num_iter, **args)
    return lgb_BO

results = bayesian_optimizer(10,10)

|   iter    |  target   | max_depth | min_da... | num_le... |
-------------------------------------------------------------
| 1         | 0.7859    | 49.98     | 128.7     | 115.4     |
| 2         | 0.7796    | 42.94     | 68.67     | 190.1     |
| 3         | 0.7784    | 28.36     | 57.48     | 107.0     |
| 4         | 0.7869    | 30.55     | 134.1     | 150.3     |
| 5         | 0.7892    | 36.73     | 162.4     | 187.3     |
| 6         | 0.7866    | 31.26     | 195.5     | 165.7     |
| 7         | 0.7804    | 37.51     | 75.45     | 175.9     |
| 8         | 0.7848    | 45.02     | 123.7     | 153.1     |
| 9         | 0.7842    | 33.06     | 124.2     | 109.2     |
| 10        | 0.7884    | 39.39     | 164.4     | 142.8     |
| 11        | 0.7882    | 39.39     | 165.1     | 143.5     |
| 12        | 0.7881    | 50.0      | 179.0     | 100.0     |
| 13        | 0.7895    | 50.0      | 200.0     | 200.0     |
| 14        | 0.7892    | 25.0      | 186.1     | 200.0     |
| 15        | 0.7882    | 25.0      | 165.9     | 113.4     |
| 16        | 0.7876    | 25.0      | 138.8     | 200.0     |
| 17        | 0.7874    | 50.0      | 169.2     | 200.0     |
| 18        | 0.7895    | 25.0      | 200.0     | 100.0     |
| 19        | 0.7893    | 49.78     | 198.9     | 120.6     |
| 20        | 0.7892    | 26.53     | 198.9     | 121.2     |
=============================================================
```

### Train Iterations with the optimized parameters
```python
def lgb_train(num_leaves, max_depth,  min_data_in_leaf):
    params = {
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "learning_rate": 0.5,
        'min_data_in_leaf': int(min_data_in_leaf),
        "force_col_wise": True,
        'verbose': -1,
        "metric": "auc",
        "objective": "binary",
    }

    x_train, x_val, y_train, y_val = train_test_split(
        train_X_scaled, train_y, test_size=0.2, random_state=3)
    lgtrain = lightgbm.Dataset(x_train, y_train)
    lgvalid = lightgbm.Dataset(x_val, y_val)
    model = (lightgbm.train(params, lgtrain, 200, valid_sets=[lgvalid],
                            early_stopping_rounds=200, verbose_eval=False))
    prediction_val = model.predict(
        test_X_scaled, num_iteration=model.best_iteration)
    return prediction_val, model
```

## Optimize Simulation Results
```python
# 5 runs of the prediction model and get mean values.
optimized_params = results.max['params']
prediction_val1, _ = lgb_train(**optimized_params)
prediction_val2, _ = lgb_train(**optimized_params)
prediction_val3, _ = lgb_train(**optimized_params)
prediction_val4, _ = lgb_train(**optimized_params)
prediction_val5, model = lgb_train(**optimized_params)
y_pred = ((prediction_val1 + prediction_val2 +b
           prediction_val3 + prediction_val4 +
           prediction_val5)/5)
df_result = pd.DataFrame(
    {'Radiant_Win_Probability': y_pred})
df_result.sort_values(by='Radiant_Win_Probability', ascending=False).head()
```
<div>
<table>
  <tr>
    <th>Match ID Hash</th>
    <th>Radiant Win Probability</th>
  </tr>
  <tr>
    <td>895</td>
    <td>0.993668</td>
  </tr>
  <tr>
    <td>1617</td>
    <td>0.990840</td>
  </tr>
  <tr>
    <td>1752</td>
    <td>0.989719</td>
  </tr>
  <tr>
    <td>245</td>
    <td>0.987856</td>
  </tr>
  <tr>
    <td>2742</td>
    <td>0.986903</td>
  </tr>
</table>
</div>

```python
feature_importance = (pd.DataFrame({'feature': train_X.columns,
                                   'importance': model.feature_importance()})
                      .sort_values('importance', ascending=False))
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance.importance,
            y=feature_importance.feature, palette=("Blues_d"))
plt.show()
```
<img src='/images/dota2/27.png'>

# CONCLUSION AND RECOMMENDATION

The nature of Dota2 as a game with only one winner and one loser makes it a suitable problem for machine learning prediction. However, real-time predictions during a game can be challenging due to the need for specific information per minute. In this project, we used multiple models to determine the most suitable approach for predicting Dota2 game outcomes. Additionally, hyperparameter tuning was performed to optimize model performance, with Bayesian optimization being the preferred method due to its efficiency when dealing with expensive-to-evaluate functions like LightGBM.

While the use of Bayesian optimization for hyperparameter tuning might not be as significant for small datasets or simple models, it becomes essential when dealing with enormous datasets where grid search may not be economically feasible. Therefore, the use of Bayesian optimization can improve the efficiency of the hyperparameter search process.

To further enhance prediction accuracy, we recommend the use of time-series machine learning models to provide real-time forecasting during the game. Such models can take into account the changing dynamics of the game and provide more accurate predictions.

In conclusion, this notebook provides valuable insights into Dota2 as a growing Esport and demonstrates how different machine learning models can be used to predict game outcomes. By leveraging hyperparameter optimization techniques like Bayesian optimization and considering time-series models for real-time prediction, we can improve the accuracy of our predictions and gain a deeper understanding of Dota2 as an Esport.

# REFERENCES

[1] Dota 2. (n.d.). https://www.dota2.com/home <br>

[2] Staff, T. G. H. (2022, September 7). What Makes Dota 2 So Successful. The Game Haus. https://thegamehaus.com/dota/what-makes-dota-2-so-successful/2022/04/02/

[3] mlcourse.ai: Dota 2 Winner Prediction | Kaggle. (n.d.). https://www.kaggle.com/competitions/mlcourse-dota2-win-prediction/overview

[4] Dota 2 5v5 - Red vs Blue by dcneil on. (2013, October 9). DeviantArt. https://www.deviantart.com/dcneil/art/Dota-2-5v5-Red-vs-Blue-406091855

[5] mlcourse.ai: Dota 2 Winner Prediction | Kaggle. (n.d.-b). https://www.kaggle.com/competitions/mlcourse-dota2-win-prediction/data
    
[6] Dota 2 Wiki. (n.d.). https://dota2.fandom.com/wiki/Dota_2_Wiki

[7] fmfn, F. (n.d.). GitHub - fmfn/BayesianOptimization: A Python implementation of global optimization with gaussian processes. GitHub. https://github.com/fmfn/BayesianOptimization   

[8] Natsume, Y. (2022, April 30). Bayesian Optimization with Python - Towards Data Science. Medium. https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec

* Note: the mltools of Prof Leodegario U. Lorenzo II and feedbacks of our Mentor Prof Gilian Uy and also the other professors significantly made this notebook very cool!







