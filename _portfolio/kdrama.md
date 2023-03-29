---
title: "Artificial K-Intelligence"
excerpt: "Using Explainable AI to Decipher the Complex World of K-Dramas<br /><img src='/images/kdrama/1.png'>"
collection: portfolio
---

<img src='/images/kdrama/1.png'>

<a name="exec_summary"></a>
<h1 style='background-color:#1BD6DD; border: 1px solid #ffffff; padding: 10px 0;'>EXECUTIVE SUMMARY</h1>

The Korean drama (KDrama) industry has witnessed a remarkable rise in global popularity, captivating audiences with its unique blend of elements. Despite this success, the industry faces significant challenges due to increasing production costs. Striking a balance between creative expression and financial constraints has become a priority for producers seeking innovative ways to create high-quality series that resonate with viewers.

By leveraging machine learning models and the SHapley Additive exPlanations (SHAP) method, producers can better navigate these challenges and develop high-quality series that appeal to their target audience.

An analysis of the TMDB KDramas 2022 dataset using AI has revealed intriguing insights into the factors influencing KDrama series' popularity. Certain themes and narratives appear to resonate more with viewers, while others garner less interest. Series based on novels or comics, exploring horror themes, or focusing on boss/employee relationships have higher appeal compared to love triangles, mini-series, and suspense-driven plots. Additionally, historical themes may negatively impact the popularity of high budget, low rating series.

These findings hold significant implications for production companies, as the selection of themes and narratives can substantially impact a KDrama series' popularity. By concentrating on elements that strongly resonate with viewers, companies can improve their chances of producing hit series with high ratings and broad appeal.

In summary, this project underscores the potential of AI and data analysis in uncovering insights into audience preferences and the factors contributing to the success of KDrama series. As the industry continues to grow and evolve, these insights will prove invaluable for producers and creators aiming to develop high-quality, engaging series that captivate viewers worldwide.

<a name="contents"></a>
<h1 style='background-color:#1BD6DD; border: 1px solid #ffffff; padding: 10px 0;'>TABLE OF CONTENTS</h1>

[ref0]: #exec_summary 
- [Executive Summary][ref0]

[ref1]: #motiv
- [Motivation][ref1]

[ref2]: #prob
- [Problem Statement][ref2]

[ref3]: #limitations
- [Limitations][ref3]

[ref4]: #metho 
- [Methodology][ref4]

[ref5]: #data
- [Data Source and Description][ref5]

[ref6]: #explo
- [Data Exploration][ref6]

[ref7]: #results
- [Machine Learning Results][ref7]

[ref8]: #explain
- [Explainability][ref8]

[ref9]: #conclu
- [Conclusion][ref9]

[ref10]: #reco
- [Recommendations][ref10]

[ref11]: #ref
- [References][ref11]


<a name="intro"></a>
<h1 style='background-color:#1BD6DD; border: 1px solid #ffffff; padding: 10px 0;'>INTRODUCTION</h1>

<a name="motiv"></a>
<h2 style='color:#01355F'>Motivation</h2>

***
The modern entertainment industry has long grappled with the challenge of **balancing creative expression and financial constraints**. As this industry grow increasingly supported yet competitive, producers have been searching for innovative solutions to produce high-quality series that **resonate with audiences**.

In particular, the Korean drama (KDrama) industry has experienced a surge in popularity in recent years, with viewers around the world captivated by its unique combination of elements. 

The allure of KDramas is multifaceted, as each element serves to create a deeply immersive and captivating viewing experience for audiences. The production value of KDramas is particularly notable, with producers investing significant resources to ensure that every detail is executed flawlessly. They have a layered storytelling and top-tier acting performances that enable viewers to connect with the characters and become fully invested in the storylines.

Moreover, their constantly evolving narrative structure of KDramas keeps viewers engaged and excited, as they never quite know what to expect. It may also be due to their catchy original soundtracks which are being featured in KDramas that add an extra layer of depth and emotion to the viewing experience, drawing viewers even further into the world of the series.

__However, the KDrama industry has been facing significant challenges in recent years the cost of producing KDrama episodes has risen to seven hundred million won (￦ 700M) by the year 2021.__ [1]

By uncovering the **critical factors** that determine the success of KDrama series and providing **actionable insights** for producers, they could make their mark in this highly competitive industry. And as a result, this may also inspire a new wave of innovation and creativity in the KDrama industry, empowering producers to bring captivating stories to life while staying true to their vision and budget.

[ref]: #contents
[Back to Table of Contents][ref]

<a name="prob"></a>
<h2 style='color:#01355F'>Problem Statement</h2>

***
With the use of artificial intelligence (AI), producers could decipher the complex world of KDrama production by can gaining insights into **how to balance creative and financial pressures while still producing high-quality series that can appeal to audiences** through the use of SHapley Additive exPlanations (SHAP) method.  

[ref]: #contents
[Back to Table of Contents][ref]

<a name="limitations"></a>
<h2 style='color:#01355F'>Limitations</h2>

***
__Many producers do not disclose the production cost information, making it difficult to obtain a comprehensive understanding of the financial pressures facing the industry.__ As a result, the team computes the production cost for this study by binning the popularity of the cast members into 3 categories _(i.e. Newbie, Experienced, Popular)_ and converting it into salaries, which were then multiplied by the number of episodes.

While this approach allowed us to estimate the production costs of KDramas with a reasonable degree of accuracy, it is important to __note that it may not fully capture the nuances of the production process__. Moreover, there may be other factors that impact the cost of producing a KDrama that we were unable to account for in our analysis.

__Another potential limitation of this study is the focus on high-budget, low-rating and low-budget, high-rating KDramas.__ While this approach allowed us to identify critical factors that contribute to the success of KDramas, it may not fully capture the complexity of the industry or the diversity of KDrama series. As such, further research is needed to explore the factors that contribute to the success of KDramas across a broader range of categories and budgets.

While our study offers valuable insights into the challenges and opportunities facing the KDrama industry, it is important to approach our findings with a degree of caution and to recognize the limitations inherent in our methodology.

[ref]: #contents
[Back to Table of Contents][ref]

<a name="metho"></a>
<h1 style='background-color:#1BD6DD; border: 1px solid #ffffff; padding: 10px 0;'>METHODOLOGY</h1>

<img src='/images/kdrama/2.png'>

**Data Extraction**

The data, TMDB KDramas 2022 dataset was taken from Kaggle and was accessed through https://www.kaggle.com/datasets/robertonacu/tmdb-kdramas-2022.

**Production Cost Estimation**

The cost of production for each cast member will be estimated based on their popularity and the salary range associated with their respective bin. These values will then be multiplied by the number of episodes in the series to estimate the total production cost for cast salaries.

**Feature Engineering**

To determine the most important features of each series, Lasso or L1 Regression was used. This step reduced the number of columns from 1,369 to 100.

Lasso Regresion or L1 Regularization was performed in order to get and retain only the most important feature for each quadrant. By doing this step, it reduces the model complexity and at the same time removes the possibility of multicollinearity on the remaining features. What Lasso does is that it adds a penalty term which is the absolute value of magnitude of coefficient to the loss function (based on the hypertuned parameter of alpha on these cases) or to laymanize, it is a technique that shrinks the less important feature’s coefficient to zero, thus useful for feature selection.


**Machine Learning Modelling**

To predict the popularity of a series, two machine learning models were used: Random Forest Classifier and used the F-1 score as its metric.

**Explanation Generation**

Interpretations of the output of the machine learning model were defined through the use of SHapley Additive exPlanations (SHAP).

[ref]: #contents
[Back to Table of Contents][ref]

<a name="data_handling"></a>
<h1 style='background-color:#1BD6DD; border: 1px solid #ffffff; padding: 10px 0;'>DATA HANDLING</h1>

<h2>Library Importation</h2>

***
```python
# helper utility function
import kdu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re, time, os, joblib
from tqdm.notebook import trange, tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from tqdm.notebook import tqdm, trange
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image as WordImage
from IPython.display import Image
from IPython.display import display, display_html, HTML
from plotnine import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, OPTICS,
                             cluster_optics_dbscan)
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

# ML
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier

# Classifier Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import (precision_score, recall_score,
                             accuracy_score, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV

# Explainability
import shap
from dice_ml import Data, Model, Dice

import warnings
warnings.filterwarnings('ignore')
```

<a name="data"></a>
<h2 style='color:#01355F'>Data Source and Description</h2>

***
The data source for this study is the TMDB KDramas 2022 dataset, which is available on Kaggle. This dataset includes information on Korean drama (KDrama) series from 1985 to 2022, as well as details on the cast, crew, and production companies involved in each series.

The dataset contains multiple files, including `series.csv`, which provides information on each KDrama series, such as its title, release year, rating, popularity, and budget. `Cast.csv` provides information on the actors and actresses who starred in each series, while `production.csv` provides information on the directors, writers, and other production crew members. `Networks.csv` lists the production companies involved in each series, while `genres.csv` provides information on the genre(s) of each series.

### Data Loading

***
```python
series_data, people_data = kdu.load_data()
kdu.df_exploration(series_data)
kdu.df_exploration(people_data)
```
<img src='/images/kdrama/3.png'>
<img src='/images/kdrama/4.png'>

