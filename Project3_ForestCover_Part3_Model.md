---
layout: page
title: Forest Cover Type - Part 3 - Ensemble Model
permalink: /project_3/
---

### Purpose:

This notebook covers development and hyper parameter tuning of an ensemble classification modele for the __Forest Cover Type Kaggle Competition__ dataset.  The goal is to build and optimized a series of standard classification models and combine them in a voting ensemble to classify the forest cover type for submission to the Kaggle Competition.

### Table of Contents

- Background: 
   + Kaggle Competition - Forest Cover Type
   + Acknowledgements
- Data Setup
- Building Individual Classification Models
  + Support Vector Machines
  + Random Forest Classifier
  + Stochastic Gradient Descent
  + k-Nearest Neighbor
- Building the Ensemble Model
- Bonus: Boosting
- Conclusions
- Next Steps

## Kaggle Competition: Forest Cover Type Documentation

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


## Data Setup

Data is first imported and engineered features are added.  (See Part 2 for feature engineering)


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


```python
#import the data
train = pd.read_csv('./datasets/train.csv')

#feature engineering
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

#preprocessing
X_train = train.drop("Cover_Type", axis=1)
y_train = train["Cover_Type"]

preprocess_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])
X_train = preprocess_pipeline.fit_transform(X_train)
```

## Building individual classification models

Start by building and optimizing several different types of classification models.  The idea here is that different types of models may be better suited at identifying some different instances of Forest Cover.  By building many types of classification models and having each one report a confidence level for the prediction, a best-of-the-best prediction can be obtained by having each model vote for the Forest Cover Type with a weighted vote value based on the confidence level of the prediction.

### Support Vector Machine


```python
#build and test baseline Support Vector Machine Model

svm_clf = SVC()
svm_clf.fit(X_train,y_train)

scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.6754629629629629



Reported score of 67.5% accuracy for the training dataset is reported above.  Now I will use the `GridSearchCV` function to optimize the model hyper-parameters to obtain the best Cross Validation score.

#### Hyper-parameter tuning

The SVM in SK-Learn has 4 kernel options: `linear`, `poly`, `rbf`, and `sigmoid`.  Each one has somewhat different hyper-parameters that are significant, so I will split out the grid-search to address each one individualy (the default used above is the `rbf` kernel)

#### Linear SVM


```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1, 0.8, 2, 10, 15, 20]}
svm = SVC(kernel='linear', cache_size=800)
svm_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=6)
svm_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=800, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=6,
           param_grid={'C': [0.1, 0.8, 2, 10, 15, 20]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
svm_search.best_params_
```




    {'C': 20}




```python
svm_search.best_score_
```




    0.6484126984126984



#### Polynomial SVM


```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.05, 0.1, 0.5, 2], 'degree':[4, 5, 7], 'coef0':[1.2, 1.5, 1.8, 2.1]}
svm = SVC(kernel='poly', cache_size=800)
svm_poly_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=7)
svm_poly_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=800, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'C': [0.05, 0.1, 0.5, 2], 'degree': [4, 5, 7], 'coef0': [1.2, 1.5, 1.8, 2.1]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
svm_poly_search.best_params_
```




    {'C': 0.1, 'coef0': 1.2, 'degree': 5}




```python
svm_poly_search.best_score_
```




    0.6659391534391534




```python
svm_poly_best = SVC(kernel='poly', cache_size=900, C=0.1, coef0=1.2, degree=5, probability=True)
svm_poly_best.fit(X_train,y_train)

scores = cross_val_score(svm_poly_best, X_train, y_train, cv=10)
scores.mean()
```




    0.6849206349206349



#### Gaussian Radial Basis Function SVM


```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[5, 10, 12, 15], 'gamma':[0.005, 0.01, 0.05], }
svm = SVC(kernel='rbf', cache_size=900)
svm_rbf_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=7)
svm_rbf_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=900, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'C': [5, 10, 12, 15], 'gamma': [0.005, 0.01, 0.05]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
svm_rbf_search.best_params_
```




    {'C': 10, 'gamma': 0.01}




```python
svm_rbf_search.best_score_
```




    0.6675925925925926




```python
svm_rbf_best = SVC(kernel='rbf', cache_size=900, C=10, gamma=0.01, probability=True)
svm_rbf_best.fit(X_train,y_train)

scores = cross_val_score(svm_rbf_best, X_train, y_train, cv=10)
scores.mean()
```




    0.6895502645502646



#### Sigmoid SVM


```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1, 2, 12, 20, 30], 'coef0':[0.0, 0.5, 1.0, 1.5]}
svm = SVC(kernel='sigmoid', cache_size=900)
svm_sig_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=7)
svm_sig_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=900, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='sigmoid',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'C': [0.1, 2, 12, 20, 30], 'coef0': [0.0, 0.5, 1.0, 1.5]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
svm_sig_search.best_params_
```




    {'C': 0.1, 'coef0': 0.0}




```python
svm_sig_search.best_score_
```




    0.5892857142857143



The best kernel is a bit of a wash between `linear`, `poly`, and `rbf` 



### Random Forest Classifier

The `RandomForestClassifier` function builds an ensemble model of its own.  This is a series of Decision Tree models by sampling the training dataset with replacement and then averages the prediction


```python
#build and test baseline Random Forest Model

forest_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.7125



Reported score of 67.5% accuracy for the training dataset is reported above.  Again applying the `GridSearchCV` function to optimize the model hyper-parameters to obtain the best Cross Validation score.

#### Hyper-parameter tuning



```python
param_grid = {'n_estimators': [60, 65, 70, 75, 80, 90], 'criterion':['gini','entropy']}

forest = RandomForestClassifier(random_state=42)
forest_search = GridSearchCV(forest, param_grid, cv=5, n_jobs=7)
forest_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=42, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'n_estimators': [60, 65, 70, 75, 80, 90], 'criterion': ['gini', 'entropy']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
forest_search.best_params_
```




    {'criterion': 'gini', 'n_estimators': 65}




```python
forest_search.best_score_
```




    0.7212962962962963




```python
forest_best = forest_search.best_estimator_
```

### Stochastic Gradient Descent


```python
from sklearn.linear_model.stochastic_gradient import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, max_iter=5, tol=None)
scores = cross_val_score(sgd_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.5328042328042328



Reported score of 53.3% accuracy for the training dataset is reported above.  Again applying the `GridSearchCV` function to optimize the model hyper-parameters to obtain the best Cross Validation score.

#### Hyper-parameter tuning



```python
from sklearn.model_selection import GridSearchCV

param_grid = {'loss': ('hinge','log','modified_huber','squared_hinge'), 
              'penalty': ('l2', 'l1','elasticnet'),
             'max_iter':[45, 50, 55, 60, 65]}
sgd = SGDClassifier(random_state=42, tol=None)
sgd_search = GridSearchCV(sgd, param_grid, cv=5, n_jobs=7)
sgd_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
           eta0=0.0, fit_intercept=True, l1_ratio=0.15,
           learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
           n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
           tol=None, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge'), 'penalty': ('l2', 'l1', 'elasticnet'), 'max_iter': [45, 50, 55, 60, 65]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
sgd_search.best_params_
```




    {'loss': 'log', 'max_iter': 60, 'penalty': 'l1'}




```python
sgd_search.best_score_
```




    0.6086640211640212




```python
sgd_best = sgd_search.best_estimator_
```

### k-Nearest Neighbor


```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
scores = cross_val_score(knn_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.6466269841269842



Reported score of 53.3% accuracy for the training dataset is reported above.  Again applying the `GridSearchCV` function to optimize the model hyper-parameters to obtain the best Cross Validation score.

#### Hyper-parameter tuning



```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [2, 4, 5, 8, 10, 15, 20], 'weights': ('uniform','distance'), 
             'algorithm': ('ball_tree','kd_tree','brute')}
knn = KNeighborsClassifier()
knn_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=7)
knn_search.fit(X_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform'),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'n_neighbors': [2, 4, 5, 8, 10, 15, 20], 'weights': ('uniform', 'distance'), 'algorithm': ('ball_tree', 'kd_tree', 'brute')},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
knn_search.best_params_
```




    {'algorithm': 'ball_tree', 'n_neighbors': 10, 'weights': 'distance'}




```python
knn_search.best_score_
```




    0.6484788359788359




```python
knn_best = knn_search.best_estimator_
```

## Building the Ensemble Model


```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('svm_rbf', svm_rbf_best), ('knn', knn_best), ('svm_poly', svm_poly_best),
                                         ('sgd', sgd_best), ('forest', forest_best)],
                             voting='soft')

voting_clf.fit(X_train, y_train)
scores = cross_val_score(voting_clf, X_train, y_train, cv=10)
scores.mean()
```


    0.6996693121693122




```python
scores.max()
```




    0.7876984126984127



## Bonus: Boosting

An added approach is to apply 'Boosting' as part of an ensemble method.  This successivly creates decision tree models by "boosting" the weight of instances that are misclassified by the previous model.


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=200,
                            algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_train, y_train)

scores = cross_val_score(ada_clf, X_train, y_train, cv=10)
scores.mean()
```




    0.6981481481481481




```python
### Optimize AdaBoost

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [150, 170, 190, 220],
             'learning_rate': [0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85]}

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), algorithm='SAMME.R')
ada_search = GridSearchCV(ada, param_grid, cv=5, n_jobs=7)
ada_search.fit(X_train, y_train)


```




    GridSearchCV(cv=5, error_score='raise',
           estimator=AdaBoostClassifier(algorithm='SAMME.R',
              base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'),
              learning_rate=1.0, n_estimators=50, random_state=None),
           fit_params=None, iid=True, n_jobs=7,
           param_grid={'n_estimators': [150, 170, 190, 220], 'learning_rate': [0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
ada_search.best_params_
```




    {'learning_rate': 0.7, 'n_estimators': 220}




```python
ada_search.best_score_
```




    0.6891534391534392



## Conclusions

The random forest enemble model on its own produced an average cross-validation prediction accuracy of 72%, this was not improved on by adding SVM, kNN, and Gradient Decent models into a larger voting ensemble model, which achieved 70% accuracy.

## Next Steps

Further improvement could be gained by developing an ensemble model using a series of 'boosted' models.  Insight on each model's performance using a confusion matrix could help prioritize voting between models.  And, ANN or DNN models could be added.
