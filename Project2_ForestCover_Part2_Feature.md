---
layout: page
title: Forest Cover Type - Part 2 - Feature Engineering
permalink: /project_2/
---


### Purpose:

This notebook covers basic classification modeling and optimization based on feature engineering for the __Forest Cover Type Kaggle Competition__ dataset.  The goal is to prepare data for modeling by enhancing the feature set with contrived attributes that improve predictive accuracy of basic Decision Tree or Support Vector Machine models.

### Table of Contents

- Background: 
   + Kaggle Competition - Forest Cover Type
   + Acknowledgements
- Data Import
- Build a Baseline Model
- Generate New Features
- Conclusions
- Next Steps

## Kaggle Competition: Forest Cover Type 

#### Competition Description:

>"In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from cartographic variables. The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type." [Kaggle Competition Page](https://www.kaggle.com/c/forest-cover-type-kernels-only)

#### Dataset Description:

>"The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:
>
>1 - Spruce/Fir
>2 - Lodgepole Pine
>3 - Ponderosa Pine
>4 - Cottonwood/Willow
>5 - Aspen
>6 - Douglas-fir
>7 - Krummholz
>
>The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).
Data Fields
>
>Elevation - Elevation in meters
Aspect - Aspect in degrees azimuth
Slope - Slope in degrees
Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation
>
>The wilderness areas are:
>
>1. - Rawah Wilderness Area
>2. - Neota Wilderness Area
>3. - Comanche Peak Wilderness Area
>4. - Cache la Poudre Wilderness Area
>
>The soil types are:
>
>1. Cathedral family - Rock outcrop complex, extremely stony.
>2. Vanet - Ratake families complex, very stony.
>3. Haploborolis - Rock outcrop complex, rubbly.
>4. Ratake family - Rock outcrop complex, rubbly.
>5. Vanet family - Rock outcrop complex complex, rubbly.
>6. Vanet - Wetmore families - Rock outcrop complex, stony.
>7. Gothic family.
>8. Supervisor - Limber families complex.
>9. Troutville family, very stony.
>10. Bullwark - Catamount families - Rock outcrop complex, rubbly.
>11. Bullwark - Catamount families - Rock land complex, rubbly.
>12. Legault family - Rock land complex, stony.
>13. Catamount family - Rock land - Bullwark family complex, rubbly.
>14. Pachic Argiborolis - Aquolis complex.
>15. unspecified in the USFS Soil and ELU Survey.
>16. Cryaquolis - Cryoborolis complex.
>17. Gateview family - Cryaquolis complex.
>18. Rogert family, very stony.
>19. Typic Cryaquolis - Borohemists complex.
>20. Typic Cryaquepts - Typic Cryaquolls complex.
>21. Typic Cryaquolls - Leighcan family, till substratum complex.
>22. Leighcan family, till substratum, extremely bouldery.
>23. Leighcan family, till substratum - Typic Cryaquolls complex.
>24. Leighcan family, extremely stony.
>25. Leighcan family, warm, extremely stony.
>26. Granile - Catamount families complex, very stony.
>27. Leighcan family, warm - Rock outcrop complex, extremely stony.
>28. Leighcan family - Rock outcrop complex, extremely stony.
>29. Como - Legault families complex, extremely stony.
>30. Como family - Rock land - Legault family complex, extremely stony.
>31. Leighcan - Catamount families complex, extremely stony.
>32. Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
>33. Leighcan - Catamount families - Rock outcrop complex, extremely stony.
>34. Cryorthents - Rock land complex, extremely stony.
>35. Cryumbrepts - Rock outcrop - Cryaquepts complex.
>36. Bross family - Rock land - Cryumbrepts complex, extremely stony.
>37. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
>38. Leighcan - Moran families - Cryaquolls complex, extremely stony.
>39. Moran family - Cryorthents - Leighcan family complex, extremely stony.
>40. Moran family - Cryorthents - Rock land complex, extremely stony."

### Acknowledgements:

The work in this notebook was created by Mark Luckeroth with guidance and examples from:

* __"Hands-On Machine Learning with Scikit-Learn and Tensorflow"__ by **Aurelien Geron**
* __"Feature engineering on forest cover type data with ensemble of decision tress"__ by **Pruthvi H.R., et al.** [https://ieeexplore.ieee.org/document/7154873/]


## Data import


```python
#library imports and setup

import numpy as np
import os
import pandas as pd
import seaborn as sns


# to make this notebook's output stable across runs
np.random.seed(42)

# for plots
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# for models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
```

## Build Baseline Model

Baseline model is used to assess the value of the added features that are created in the next section


```python
#import the data
train = pd.read_csv('./datasets/train.csv')
X_train = train.drop("Cover_Type", axis=1)
y_train = train["Cover_Type"]

preprocess_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])
X_train = preprocess_pipeline.fit_transform(X_train)

#build and test baseline Support Vector Machine Model

svm_clf = SVC()
svm_clf.fit(X_train,y_train)

scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.6453703703703704




```python
#Optimize hyperparameters for baseline SVM model

from sklearn.model_selection import GridSearchCV

param_grid = {'kernel': ('linear', 'poly', 'sigmoid', 'rbf'), 'C':[0.1, 0.5, 0.8, 1, 2, 5, 10]}
svm = SVC()
svm_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=3)
svm_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=3,
           param_grid={'kernel': ('linear', 'poly', 'sigmoid', 'rbf'), 'C': [0.1, 0.5, 0.8, 1, 2, 5, 10]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
svm_search.best_params_
```




    {'C': 10, 'kernel': 'rbf'}




```python
svm_search.best_score_
```




    0.6426587301587302




```python
#build and test baseline Random Forest Model

forest_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.6539021164021163




```python
#Optimize hyperparameters for Random Forest Model

param_grid = {'n_estimators': [5, 8, 10, 12, 15, 20], 'criterion': ('gini','entropy')}

forest = RandomForestClassifier(random_state=42)
forest_search = GridSearchCV(forest, param_grid, cv=5, n_jobs=3)
forest_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=42, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=3,
           param_grid={'n_estimators': [5, 8, 10, 12, 15, 20], 'criterion': ('gini', 'entropy')},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
forest_search.best_params_
```




    {'criterion': 'gini', 'n_estimators': 15}




```python
forest_search.best_score_
```




    0.6390873015873015




```python
forest_best = forest_search.best_estimator_
```


```python
plt.figure(figsize=(20,20))
plt.barh(train.drop("Cover_Type", axis=1).columns.values, forest_best.feature_importances_)
plt.title('Feature Importance')
plt.ylabel('Feature Name')
plt.xlabel('Gini Value')
plt.show()
```

![png]({{site.url}}/assets/Project2_ForestCover_Part2_Feature_files/Forest%20Cover%20Type%20-%20Feature%20Engineering_16_0.png)


## Generate New Features


```python
train['shade_sum'] = train[['Hillshade_3pm','Hillshade_Noon','Hillshade_9am']].sum(axis=1)
train['gwtr_lin'] = train['Vertical_Distance_To_Hydrology']+train['Horizontal_Distance_To_Hydrology']
train['gwtr_quad'] = train['Vertical_Distance_To_Hydrology']+(train['Horizontal_Distance_To_Hydrology']**2)
train['gwtr_euclid'] = ((train['Vertical_Distance_To_Hydrology']**2) + (train['Horizontal_Distance_To_Hydrology']**2))**0.5
train['fir_wtr'] = train['Horizontal_Distance_To_Fire_Points']+train['gwtr_lin']
train['fir_rd'] = train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways']
train['elev_x_slop'] = train['Elevation'] + train['Slope']
train['elev_x_asp'] = train['Elevation'] + train['Aspect']
train['elev_m_slop'] = train['Elevation'] - train['Slope']
train['elev_m_asp'] = train['Elevation'] - train['Aspect']
train['fir_wtr_diff'] = train['Horizontal_Distance_To_Fire_Points']-train['gwtr_lin']
train['fir_rd_diff'] = train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways']
train['hot_sun'] = train['Hillshade_3pm']+train['Hillshade_Noon'] - train['Hillshade_9am']
train['cool_sun'] = train['Hillshade_9am']-train['Hillshade_Noon'] - train['Hillshade_3pm']

X_train = train.drop("Cover_Type", axis=1)
y_train = train["Cover_Type"]

preprocess_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])
X_train = preprocess_pipeline.fit_transform(X_train)

```


```python
#build and test baseline Random Forest Model

forest_clf = RandomForestClassifier(random_state=42, criterion='gini', n_estimators=15)
forest_clf.fit(X_train, y_train)
scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.7234788359788359




```python
plt.figure(figsize=(20,20))
plt.barh(train.drop("Cover_Type", axis=1).columns.values, forest_clf.feature_importances_)
plt.title('Feature Importance')
plt.ylabel('Feature Name')
plt.xlabel('Gini Value')
plt.show()
```


![png]({{site.url}}/assets/Project2_ForestCover_Part2_Feature_files/Forest%20Cover%20Type%20-%20Feature%20Engineering_20_0.png)


## Conclusions

Simple linear combination of a number of the given features gives a significant boost to model accuracy for a `RandomForestClassifier` model: from 63.9% to 71.2% on the training dataset.  Further improvement can likely be obtained using non-linear combination or adding additional research.

## Next Steps

Using the enhanced dataset produced here, build an ensemble classifier model and add Boosting methods and hyper parameter optimization to obtain the final predictive model
