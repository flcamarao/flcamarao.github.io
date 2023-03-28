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