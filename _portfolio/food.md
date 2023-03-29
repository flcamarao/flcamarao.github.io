---
title: "Pressure's off the Menu! Enjoyable Plating for Healthy Living"
excerpt: "Recommender System of Delicious and Healthy Recipes for Hypertensive Eaters<br /><img src='/images/food/1.png'>"
collection: portfolio
---

<img src='/images/food/1.png'>

<a name="exec_summary"></a>

***
<h1 style="color:#F15025">EXECUTIVE SUMMARY</h1>

***
__High blood pressure, also known as hypertension, is a common chronic condition affecting millions of people worldwide.__ It is a major risk factor for cardiovascular disease, stroke, and kidney disease, among other health problems. Diet plays a crucial role in managing hypertension, and people with high blood pressure often struggle to find healthy and tasty recipes that meet their nutritional needs. [1] While there are many recipe websites and software applications available, they do not always provide tailored recommendations for people with specific health conditions such as hypertension. Moreover, __some recipes labeled as healthy may not be appropriate for people with high blood pressure due to their sodium content or other ingredients that can exacerbate the condition.__

This project focused on helping hypertensive consumers overcome the challenge of adhering to dietary recommendations. As the prevalence of this chronic disease continues to rise globally, proper diet management is crucial. However, patients often struggle to cook healthy and tasty meals that fit their nutritional limitations. To address this problem, our team developed a personalized meal recommender system that takes into account consumers' nutritional requirements, taste preferences, cooking skills, and other relevant factors. The goal is to enhance the user experience of hypertensive consumers and empower them to make informed and enjoyable food choices.

`Recipes`, `User Ratings`, and `nutritional values` were scraped and extracted from `allrecipes.com`[2]. Post pre-processing, the recipes, which includes selected ingredients and its corresponding nutritional values, were then vectorized for Term Frequency-Inverse Document Frequency (TF-IDF). Singular Value Decomposition (SVD) was also applied to help identify latent features that are relevant for recipe recommendations.
Finally, using K-means clustering method, the recipes were then grouped and labelled as either `Sweet Treats` or `Savory Treats` based on the cluster features. Using Neighborhood-based Collaborative Filtering recommendation system method, the algorithm returned recipes that fits the consumers preference. The recommended items were further filtered based on the 2,000 grams sodium threshold as prescribed by WHO for hypertensive individuals. [3]

The team is confident in the potential of this recommendation system, and we have identified several key areas for improvement and expansion such as, but not limited to the following: (1) enhance the customization capturing consumer specific dietary requirements (2) extend the explainability of the recommended recipes to eaters, (3) expand the recommendation system to include other health conditions such as diabetes, cancer, gout, and others, and (4) maximize the impact and reach of our recommendation system through partnerships with food delivery services.

Overall, this project highlighted the potential of data-driven approaches to promote healthier eating habits, especially for those with specific health concerns such as hypertension. **By leveraging machine learning techniques and nutritional expertise, we can create personalized recommendations that are both enjoyable and beneficial for individuals' health.**

[ref]: #top
[Back to Table of Contents][ref]

<a name="top"></a>

***
<h1 style="color:#F15025">TABLE OF CONTENTS</h1>

***

[ref0]: #exec_summary 
- [Executive Summary][ref0]

[ref1]: #prob_statement 
- [Problem Statement][ref1]

[ref2]: #motivation 
- [Motivation][ref2]

[ref3]: #list 
- [List of Tables and Figures][ref3]

[ref4]: #methodology 
- [Methodology][ref4]

[ref5]: #source 
- [Data Source and Description][ref5]

[ref6]: #exploration
- [Data Exploration][ref6]

[ref7]: #dimension
- [Dimensionality Reduction][ref7]

[ref8]: #results
- [Results][ref8]

[ref9]: #conclusion
- [Conclusion][ref9]

[ref10]: #recommendation
- [Recommendations][ref10]

[ref11]: #references
- [References][ref11]

<a name="prob_statement"></a>

***
<h1 style="color:#F15025">PROBLEM STATEMENT</h1>

***
How the world eats is shifting dramatically. The proliferation of different food options alongside technological advancements increased convenience but at the same time lessened consumers’ nutritional awareness. Given this objective in mind, we explored information from `allrecipes` focusing on this key question that our team seeks to address:

__How can we recommend delicious and healthy recipes to promote the well-being of hypertensive eaters?__

The team's primary focus is on developing a recommendation system that can effectively suggest tasty and nutritious recipes to individuals with hypertension. This question addresses the main challenge of balancing the dietary restrictions required to manage hypertension while still promoting enjoyable and satisfying meals.

Several key considerations such as identifying and incorporating relevant nutritional guidelines, capturing user preferences and dietary restrictions, and developing a robust and accurate recommendation algorithm were taken into consideration. _However, for the time being, the team will only incorporate the World Health Organization's 2,000mg daily sodium restriction as the nutritional guideline._ By addressing these challenges, the team aims to provide an effective solution to promote the well-being of hypertensive individuals while enjoying their meals.

[ref]: #top
[Back to Table of Contents][ref]

<a name="motivation"></a>

***
<h1 style="color:#F15025">MOTIVATION</h1>

***

__Hypertension, or high blood pressure, is a common medical condition that affects millions of people worldwide, including many of our loved ones.__ It is a major risk factor for cardiovascular disease, stroke, and other serious health problems, and is a leading cause of premature death globally.

One of the main ways to manage hypertension is through lifestyle modifications, including changes to diet and exercise habits. __For many hypertensive individuals, this means adhering to specific dietary restrictions, such as reducing salt intake and increasing consumption of nutrient-rich foods like fruits, vegetables, and whole grains.__

However, following these dietary restrictions can be challenging for many patients, especially if they feel that they have to sacrifice flavor or enjoyment in their meals. This is where the team aims to help hypertensive patients come in. __By developing a recommendation system that suggests delicious and healthy recipes tailored to their dietary requirements, our team hopes to make it easier for hypertensive individuals to adhere to their restrictions while still enjoying their meals.__

This can have significant health benefits for hypertensive patients, as it can help them to maintain a healthy weight, manage their blood pressure, and reduce their risk of developing other health problems associated with hypertension. __Overall, the team aims to improve the quality of life and health outcomes for hypertensive individuals by helping them to make tasty and nutritious food choices.__

[ref]: #top
[Back to Table of Contents][ref]

<a name="list"></a>

***
<h1 style="color:#F15025">LIST OF TABLES AND FIGURES</h1>

***

[ref12]: #table1
[**Table 1.** Description of the Data per DataFrame][ref12]

[ref13]: #table2
[**Table 2.** Features of the Recipes Dataset][ref13]

[ref14]: #table3 
[**Table 3.** Features of the Ratings Dataset][ref14]

---
[ref15]: #figure1
[**Figure 1.** Project Methodology][ref15]

[ref16]: #figure2
[**Figure 2.** Distribution of Ratings per Rating Value][ref16]

[ref17]: #figure3
[**Figure 3.** Top 10 Foods with the Most Number of Reviews][ref17]

[ref18]: #figure4
[**Figure 4.** Top 10 Foods with the Most Number of Sodium content (in mg)][ref18]

[ref19]: #figure5
[**Figure 5:** Top 10 Foods with the Most Number of Sodium content (in mg) per Serving][ref19]

[ref20]: #figure6
[**Figure 6:** Top 10 Foods with the Least Number of Sodium content (in mg) per Serving][ref20]

[ref21]: #figure7
[**Figure 7:** Cumulative Variance Explained][ref21]

[ref22]: #figure8
[**Figure 8:** Results of k-Means Clustering][ref22]

[ref23]: #figure9
[**Figure 9:** Results of k-Means Clustering Metrics][ref23]

[ref24]: #figure10
[**Figure 10:** Final k-Means Clustering][ref24]

[ref25]: #figure11
[**Figure 11:** Hierarchical Clustering Methods][ref25]

[ref26]: #figure12
[**Figure 12:** Extracted Cluster Themes][ref26]

[ref27]: #figure13
[**Figure 13:** FlavorFit Recommender System][ref27]

[ref28]: #figure14
[**Figure 14:** FlavorFit Recommender System for Sweet Treats][ref28]

[ref29]: #figure15
[**Figure 15:** FlavorFit Recommender System for Savory Eats][ref29]


[ref]: #top
[Back to Table of Contents][ref]

<a name="methodology"></a>

***
<h1 style="color:#F15025">METHODOLOGY</h1>

***
<a name="figure1"></a>
<img src='/images/food/2.png'>
<br>
<center><b>Figure 1:</b> Project Methodology </center>
To develop the proposed recipe recommendation system for hypertensive eaters, the following methodology was employed: 

__1. Data Extraction, Cleaning and Processing:__ 

_Data Extraction_

The recipes, ratings, and nutrients for this study were extracted from allrecipes.com, a popular recipe website that provides nutritional values for each recipe. 
    

_Data Cleaning & Pre-processing_

The following cleaning and preprocessing steps were performed to clean the datasets:
1. Combined the food and nutrients dataframes.
2. Retained only the calories, carbohydrates_g, sugars_g, fat_g, saturated_fat_g, cholesterol_mg, protein_g, dietary_fiber_g, sodium_mg, and calories_from_fat nutrients. Dropped the rest of the nutrients as they either contain null information or will not impact consumers diagnosed with hypertension. 
3. Retained only the ratings where the food_id is also in the food datarame. 
4. Cleaned the recipe text by employing lemmatization and removing stopwords unique to the dataset (e.g., chef, easy, make, etc).
    
    
__2. Text Vectorization and Clustering__

The resulting recipes after the data cleaning and pre-procesing stage are then vectorized for Term Frequency-Inverse Document Frequency (TF-IDF). This is to identify the most important or relevant words in a recipe (e.g., key ingredients and/or preparation methods). 
Afterwhich, Singular Value Decomposition (SVD), a dimensionality reduction technique most often used in recommender systems, was employed to help identify latent features that are relevant for recipe recommendations.   

Finally, using K-means clustering method, the recipes are grouped based on their features like ingredients, preparation, and nutritional values.

__3. Recommendation System__

Using _Neighborhood Based Collaborative Filtering method_, a machine learning algorithm that makes personalized recommendations based on the preferences and behaviours of similar users, the alogirthm returned recipes that fits the eaters preference in foods. This is then filtered by a 2,000 gram sodium threshold as prescribed by WHO. 

[ref]: #top
[Back to Table of Contents][ref]

<a name="source"></a>

***
<h1 style="color:#F15025">DATA SOURCE AND DESCRIPTION</h1>

***
For this project, the team focused on exploring the Recipes and Ratings datasets with detailed features as follows:
<a name="table1"></a>

|DataFrame | Feature | Data Type | Description |
|:--------|:----------|:----------|:----------|
| `food`| `itemid` | int64 | ItemID of the recipe |
| `food` | `title`| object | Title of the recipe |
| `food` | `prep_time`| int64 | Preperation time for the recipe |
| `food` | `cook_time`| int64 | Cooking time for the recipe |
| `food` | `ready_time`| int64 | Readying time for the recipe |
| `food` | `ingredients`| object | Ingredients of the recipe |
| `food` | `directions`| object | Directions of the recipe |
| `food` | `url`| object | URL of the recipe |
| `food` | `photo_url`| object | Photo URL of the recipe |
| `nutrients` | `name`| object | Name of the recipe |
| `nutrients` | `url`| object | URL of the recipe |
| `nutrients` | `category`| object | Category of the recipe |
| `nutrients` | `author`| object | Author of the recipe |
| `nutrients` | `summary`| object | Summary of the recipe |
| `nutrients` | `rating`| float64 | Rating per recipe |
| `nutrients` | `rating_count`| int64 |Rating Count per recipe |
| `nutrients` | `review_count`| int64 | Review Count per recipe |
| `nutrients` | `ingredients`| object | Ingredients of the recipe|
| `nutrients` | `directions`| object | Directions of the recipe |
| `nutrients` | `prep`| object | Preparation of the recipe |
| `nutrients` | `cook`| object | Cooking time of each step of the recipe |
| `nutrients` | `total`| object | Total Cooking Time of the recipe |
| `nutrients` | `servings`| int64 | Serving size of the recipe |
| `nutrients` | `yield`| object | Yield of the recipe |
| `nutrients` | `calories`| float64 | Total Calories of the recipe |
| `nutrients` | `carbohydrates_g`| float64 | Total Carbohydrates in grams of the recipe |
| `nutrients` | `sugars_g`| float64 | Total Sugar in grams of the recipe |
| `nutrients` | `fat_g`| float64 | Total Fat in grams of the recipe |
| `nutrients` | `saturated_fat_g`| float64 | Total Saturated Fat in grams of the recipe |
| `nutrients` | `cholesterol_mg`| float64 | Total Cholesterol in milligrams of the recipe |
| `nutrients` | `dietary_fiber_g`| float64 | Total Dietary fiber in grams of the recipe |
| `nutrients` | `sodium_mg`| float64 | Total Sodium in milligrams of the recipe |
| `nutrients` | `calories_from_fat`| float64 | Total Calories from fat of the recipe |
| `nutrients` | `calcium_mg`| float64 | Total Calcium in milligrams of the recipe |
| `nutrients` | `iron_mg`| float64 | Total Iron in milligrams of the recipe |
| `nutrients` | `magnesium_mg`| float64 | Total Magnesium in milligrams of the recipe |
| `nutrients` | `potassium_mg`| float64 | Total Potassium in milligrams of the recipe |
| `nutrients` | `zinc_mg`| float64 | Total Zinc in milligrams of the recipe |
| `nutrients` | `phosphorus_mg`| float64 | Total Vitamin A of the recipe |
| `nutrients` | `vitamin_a_iu_IU`| float64 | Total Phosphorous in milligrams of the recipe |
| `nutrients` | `niacin_equivalents_mg`| float64 | Total Niacin Equivalents in milligrams of the recipe |
| `nutrients` | `vitamin_b6_mg`| float64 | Total Vitamin B6 in milligrams of the recipe |
| `nutrients` | `vitamin_c_mg`| float64 | Total Vitamin C in milligrams of the recipee |
| `nutrients` | `folate_mcg`| float64 | Total Folate in microgram of the recipe |
| `nutrients` | `thiamin_mg`| float64 | Total Thiamin in milligrams of the recipe |
| `nutrients` | `riboflavin_mg`| float64 | Total Riboflavin in milligrams of the recipe |
| `nutrients` | `vitamin_e_iu_IU`| float64 | Total Vitamin E of the recipe |
| `nutrients` | `vitamin_k_mcg`| float64 | Total Vitamin K in micrograms of the recipe |
| `nutrients` | `biotin_mcg`| float64 | Total Biotin in micrograms of the recipe |
| `nutrients` | `vitamin_b12_mcg`| float64 | Total Vitamin B12 in micrograms of the recipe |
| `nutrients` | `mono_fat_g`| float64 | Total Mono Fat in grams of the recipe |
| `nutrients` | `poly_fat_g`| float64 | Total Poly Fat in grams of the recipe |
| `nutrients` | `trans_fatty_acid_g`| float64 | Total Trans Fatty Acid in grams of the recipe |
| `nutrients` | `omega_3_fatty_acid_g`| float64 | Total Omega 3 Fatty Acid in grams of the recipe |
| `nutrients` | `omega_6_fatty_acid_g`| float64 | Total Omega 3 Fatty Acid in grams of the recipe |
| `ratings` | `User_ID`| int64 | User ID of the rater |
| `ratings` | `Food_ID`| int64 | Food ID of the recipe |
| `ratings` | `Rating`| int64 | Rating of the rater per recipe |


<center><b> Table 1.</b> Description of the Data per DataFrame</center>


From the above table and as discussed in the methodology that the recipes and nutrients were combined, this project focused on exploring the `Recipes` and `Ratings` datasets with detailed features as follows. 

- The `Recipes` dataset contains 12,351 rows and 14 columns, including detailed information on the nutritional values of each recipe. 

    For this project, we used the following nutritional values per recipe: **calories, carbohydrates_g, sugars_g, fat_g, saturated_fat_g, cholesterol_mg, protein_g, dietary_fiber_g, and sodium_mg**. These values were documented based on the ingredients used in each recipe, which were sourced from a reliable nutritional database. By incorporating these nutritional values, eaters with specific dietary needs, or even those simply interested in tracking their nutritional intake, will greatly benefit from this. More specifically, by including the sodium content, eaters with hypertension can manage their condition better.

    Further, adding nutritional values, including sodium content, can help consumers with hypertension manage their condition better. This information can lead to better health choices and outcomes, improved quality of life, and reduced risk of complications such as heart disease and stroke. Overall, including nutritional values in food recipes is a small but significant step towards a healthier lifestyle for everyone.


- The `Ratings` dataset contains 1,555,581 rows and 3 columns, including User_id and Rating information on each recipe given by users. 

    In order to increase the reliability of the reviews and ensure that they were written by knowledgeable users, we filtered the dataset to only include reviews written by users who had provided feedback more than 100 times. By doing so, we were able to reduce the influence of potentially biased or unreliable reviews, and focus on the feedback provided by experienced reviewers. This filtering process helped us to ensure the overall quality and accuracy of the reviews used in this analysis.

By analyzing these datasets, we aim to gain insights into the nutritional value of popular recipes, identify trends in the types of ingredients and preparation methods used by allrecipes.com users, as well as understand the ratings and user behavior. In the following sections, we will describe the data analysis process and present our findings.
<a name="table2"></a>
<br>
<table>
  <thead>
    <center style="font-size:12px;font-style:default;"><b>Table 2. Features of the Recipe Dataset</b></center>
    <tr>
      <th style="text-align:center">Features</th>
      <th style="text-align:center">Data Type</th>       
      <th style="text-align:center">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left">Food_ID</td>
      <td style="text-align:left">int</td>        
      <td style="text-align:left">unique identifier of the food item</td>
    </tr>
    <tr>
      <td style="text-align:left">name</td>
      <td style="text-align:left">string</td>        
      <td style="text-align:left">name of the food item</td>
    </tr>
    <tr>
      <td style="text-align:left">ingredients</td>
      <td style="text-align:left">string</td>                
      <td style="text-align:left">list of ingredients</td>
    </tr>
    <tr>
      <td style="text-align:left">photo_url</td>
      <td style="text-align:left">string</td>        
      <td style="text-align:left">URL to a photo of the food item</td>
    </tr>
    <tr>
      <td style="text-align:left">calories</td>
      <td style="text-align:left">float</td>        
      <td style="text-align:left">total number of calories</td>
    </tr>
    <tr>
      <td style="text-align:left">carbohydrates_g</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total number of carbohydrates in grams</td>
    </tr>
    <tr>
      <td style="text-align:left">sugars_g</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total number of sugars in grams</td>
    </tr>
    <tr>
      <td style="text-align:left">fat_g</td>
      <td style="text-align:left">float</td>      
      <td style="text-align:left">total number of fat in grams</td>
    </tr>
    <tr>
      <td style="text-align:left">saturated_fat_g</td>
      <td style="text-align:left">float</td>   
      <td style="text-align:left">total number of saturated fat in grams</td>
    </tr>
    <tr>
      <td style="text-align:left">cholesterol_mg</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total amount of cholesterol in milligrams</td>
    </tr>
    <tr>
      <td style="text-align:left">protein_g</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total number of protein in grams</td>
    </tr>
    <tr>
      <td style="text-align:left">dietary_fiber_g</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total number of dietary fiber in grams</td>
    </tr>
    <tr>
      <td style="text-align:left">sodium_mg</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total amount of sodium in milligrams</td>
    </tr>
    <tr>
      <td style="text-align:left">calories_from_fat</td>
      <td style="text-align:left">float</td>    
      <td style="text-align:left">total number of calories from fat</td>
    </tr>
  </tbody>
</table>

<a name="table3"></a>

<table>
  <thead>
    <tr>
    <center style="font-size:12px;font-style:default;"><b>Table 3. Features of the Ratings Dataset</b></center>
    </tr>
    <tr>
      <th style="text-align:center">Features</th>
      <th style="text-align:center">Data Type</th>     
      <th style="text-align:center">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left">Food_ID</td>
      <td style="text-align:left">int</td>        
      <td style="text-align:left">unique identifier for the food item</td>
    </tr>
    <tr>
      <td style="text-align:left">User_id</td>
      <td style="text-align:left">int</td>          
      <td style="text-align:left">unique identifier for the user who rated the food item</td>
    </tr>
    <tr>
      <td style="text-align:left">Rating</td>
      <td style="text-align:left">float</td>        
      <td style="text-align:left">rating given by the user for the food item</td>
    </tr>
  </tbody>
</table>

[ref]: #top
[Back to Table of Contents][ref]

<a name="exploration"></a>

***
<h1 style="color:#F15025">DATA EXPLORATION</h1>

***

<h2 style="color:#F7AF31">Data Collection</h2>

***

`Allrecipes.com` is a popular recipe-sharing website that has been providing home cooks with a vast collection of recipes for over two decades. One of the unique features of `allrecipes.com` is that it includes nutritional information for many of its recipes. This feature has made it a go-to source for individuals who are looking for healthier meal options or who have specific dietary needs.

In this study, the team scraped selected data from `allrecipes.com` due to its diverse food options and the availability of nutritional information for majority of its recipes. By leveraging this information, the team could provide more accurate and comprehensive information about the recipes, including nutritional values such as _calories, carbohydrates_g, sugars_g, fat_g, saturated_fat_g, cholesterol_mg, protein_g, dietary_fiber_g, and sodium_mg_. This information can benefit a wide range of users, from consumers with specific dietary needs to consumers simply interested in tracking their nutritional intake.