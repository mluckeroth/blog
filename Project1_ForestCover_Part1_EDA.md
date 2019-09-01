---
layout: page
title: Forest Cover Type - Part 1 - Exploritory Data Analysis
permalink: /project_1/
---


### Purpose:

This notebook covers basic exploritory data analysis for the __Forest Cover Type Kaggle Competition__ dataset.  The goal is to familiarize the analyst with the data, determine features that have most influnce on predictor, prepare data for modeling, and inform analyst on the best starting point for modeling 

### Table of Contents

- Background: 
   + Kaggle Competition - Forest Cover Type
   + Acknowledgements
- Data Import
- Exploritory Data Analysis and Visualization
- Conclusions
- Next Steps

## Background: Kaggle Competition - Forest Cover Type 

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
* Kaggel kernel by elitcohen [https://www.kaggle.com/elitcohen/forest-cover-type-eda-modeling-error-analysis](https://www.kaggle.com/elitcohen/forest-cover-type-eda-modeling-error-analysis)

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
```


```python
#import the data
train = pd.read_csv('./datasets/train.csv')
train.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>...</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2579</td>
      <td>132</td>
      <td>6</td>
      <td>300</td>
      <td>-15</td>
      <td>67</td>
      <td>230</td>
      <td>237</td>
      <td>140</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2606</td>
      <td>45</td>
      <td>7</td>
      <td>270</td>
      <td>5</td>
      <td>633</td>
      <td>222</td>
      <td>225</td>
      <td>138</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>2605</td>
      <td>49</td>
      <td>4</td>
      <td>234</td>
      <td>7</td>
      <td>573</td>
      <td>222</td>
      <td>230</td>
      <td>144</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>2617</td>
      <td>45</td>
      <td>9</td>
      <td>240</td>
      <td>56</td>
      <td>666</td>
      <td>223</td>
      <td>221</td>
      <td>133</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>2612</td>
      <td>59</td>
      <td>10</td>
      <td>247</td>
      <td>11</td>
      <td>636</td>
      <td>228</td>
      <td>219</td>
      <td>124</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 56 columns</p>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15120 entries, 0 to 15119
    Data columns (total 56 columns):
    Id                                    15120 non-null int64
    Elevation                             15120 non-null int64
    Aspect                                15120 non-null int64
    Slope                                 15120 non-null int64
    Horizontal_Distance_To_Hydrology      15120 non-null int64
    Vertical_Distance_To_Hydrology        15120 non-null int64
    Horizontal_Distance_To_Roadways       15120 non-null int64
    Hillshade_9am                         15120 non-null int64
    Hillshade_Noon                        15120 non-null int64
    Hillshade_3pm                         15120 non-null int64
    Horizontal_Distance_To_Fire_Points    15120 non-null int64
    Wilderness_Area1                      15120 non-null int64
    Wilderness_Area2                      15120 non-null int64
    Wilderness_Area3                      15120 non-null int64
    Wilderness_Area4                      15120 non-null int64
    Soil_Type1                            15120 non-null int64
    Soil_Type2                            15120 non-null int64
    Soil_Type3                            15120 non-null int64
    Soil_Type4                            15120 non-null int64
    Soil_Type5                            15120 non-null int64
    Soil_Type6                            15120 non-null int64
    Soil_Type7                            15120 non-null int64
    Soil_Type8                            15120 non-null int64
    Soil_Type9                            15120 non-null int64
    Soil_Type10                           15120 non-null int64
    Soil_Type11                           15120 non-null int64
    Soil_Type12                           15120 non-null int64
    Soil_Type13                           15120 non-null int64
    Soil_Type14                           15120 non-null int64
    Soil_Type15                           15120 non-null int64
    Soil_Type16                           15120 non-null int64
    Soil_Type17                           15120 non-null int64
    Soil_Type18                           15120 non-null int64
    Soil_Type19                           15120 non-null int64
    Soil_Type20                           15120 non-null int64
    Soil_Type21                           15120 non-null int64
    Soil_Type22                           15120 non-null int64
    Soil_Type23                           15120 non-null int64
    Soil_Type24                           15120 non-null int64
    Soil_Type25                           15120 non-null int64
    Soil_Type26                           15120 non-null int64
    Soil_Type27                           15120 non-null int64
    Soil_Type28                           15120 non-null int64
    Soil_Type29                           15120 non-null int64
    Soil_Type30                           15120 non-null int64
    Soil_Type31                           15120 non-null int64
    Soil_Type32                           15120 non-null int64
    Soil_Type33                           15120 non-null int64
    Soil_Type34                           15120 non-null int64
    Soil_Type35                           15120 non-null int64
    Soil_Type36                           15120 non-null int64
    Soil_Type37                           15120 non-null int64
    Soil_Type38                           15120 non-null int64
    Soil_Type39                           15120 non-null int64
    Soil_Type40                           15120 non-null int64
    Cover_Type                            15120 non-null int64
    dtypes: int64(56)
    memory usage: 6.5 MB


From `train.info()` we can see there is no missing or corrupt data points, the data is clean and *"tidy"* to start with.  From `train.head(10)` we can see that all the categorical features are 'onehot' encoded, this simplifies preparing the dataset for modeling, but will need to be reversed to use some data visualization libraries below


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>...</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15120.00000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>...</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7560.50000</td>
      <td>2749.322553</td>
      <td>156.676653</td>
      <td>16.501587</td>
      <td>227.195701</td>
      <td>51.076521</td>
      <td>1714.023214</td>
      <td>212.704299</td>
      <td>218.965608</td>
      <td>135.091997</td>
      <td>...</td>
      <td>0.045635</td>
      <td>0.040741</td>
      <td>0.001455</td>
      <td>0.006746</td>
      <td>0.000661</td>
      <td>0.002249</td>
      <td>0.048148</td>
      <td>0.043452</td>
      <td>0.030357</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4364.91237</td>
      <td>417.678187</td>
      <td>110.085801</td>
      <td>8.453927</td>
      <td>210.075296</td>
      <td>61.239406</td>
      <td>1325.066358</td>
      <td>30.561287</td>
      <td>22.801966</td>
      <td>45.895189</td>
      <td>...</td>
      <td>0.208699</td>
      <td>0.197696</td>
      <td>0.038118</td>
      <td>0.081859</td>
      <td>0.025710</td>
      <td>0.047368</td>
      <td>0.214086</td>
      <td>0.203880</td>
      <td>0.171574</td>
      <td>2.000066</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1863.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-146.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>99.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3780.75000</td>
      <td>2376.000000</td>
      <td>65.000000</td>
      <td>10.000000</td>
      <td>67.000000</td>
      <td>5.000000</td>
      <td>764.000000</td>
      <td>196.000000</td>
      <td>207.000000</td>
      <td>106.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7560.50000</td>
      <td>2752.000000</td>
      <td>126.000000</td>
      <td>15.000000</td>
      <td>180.000000</td>
      <td>32.000000</td>
      <td>1316.000000</td>
      <td>220.000000</td>
      <td>223.000000</td>
      <td>138.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11340.25000</td>
      <td>3104.000000</td>
      <td>261.000000</td>
      <td>22.000000</td>
      <td>330.000000</td>
      <td>79.000000</td>
      <td>2270.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>167.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15120.00000</td>
      <td>3849.000000</td>
      <td>360.000000</td>
      <td>52.000000</td>
      <td>1343.000000</td>
      <td>554.000000</td>
      <td>6890.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>248.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 56 columns</p>
</div>



## Exploritory Data Analysis and Visualization


```python
soils = ['Soil_Type'+str(i+1) for i in list(range(40))]
areas = ['Wilderness_Area'+str(i+1) for i in list(range(4))]
train_quant = train.drop((soils + areas + ['Id','Cover_Type']), axis=1)
train_quant.hist(bins=50, figsize=(20,15))
plt.show()
```


![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_11_0.png)


The distribution of quantitative feature measurements by themselves shown above.  


```python
#convert onehot soil and wilderness data into labels

def get_soil(row):
    for c in soils:
        if row[c]==1:
            return c
        
def get_area(row):
    for c in areas:
        if row[c]==1:
            return c
        
soil_labels = train.apply(get_soil, axis=1)
area_labels = train.apply(get_area, axis=1)
labels_df = pd.DataFrame({'Soil': soil_labels, 'Area': area_labels})
train_labels = pd.concat([train, labels_df], axis=1)
```


```python
plt.subplots(figsize=(12,6))
ax = sns.countplot(x="Area", hue="Cover_Type", data=train_labels, order=areas).set_title("Cover Type distribution by Wilderness Area")

```
![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_14_0.png)



Above the distribution of Cover Types are shown for each Wilderness Area.  The Wilderness_Area appears to have a strong indication on which Cover Types are likely to occur.  Wilderness_Area4 (Cache la Poudre Wilderness Area) is the only area to have Cover_Type4 (Cottonwood/Willow).  Wilderness Areas 1,2 & 4 (Rawah Wilderness Area, Neota Wilderness Area, & Cache la Poudre Wilderness Area) all reduce the possible Cover Types to just three or four types.  Wilderness_Area3 (Comanche Peak Wilderness Area) offers the least information by only eliminating Cover_Type4 (Cottonwood/Willow), and the remaining Cover Types being of similar likelyhood.


```python
corr = train.corr()
corr["Cover_Type"].sort_values(ascending=False)
```




    Cover_Type                            1.000000
    Soil_Type38                           0.257810
    Soil_Type39                           0.240384
    Soil_Type40                           0.205851
    Soil_Type10                           0.128972
    Wilderness_Area3                      0.122146
    Soil_Type35                           0.114327
    Id                                    0.108363
    Slope                                 0.087722
    Wilderness_Area4                      0.075774
    Vertical_Distance_To_Hydrology        0.075647
    Soil_Type37                           0.071210
    Soil_Type17                           0.042453
    Soil_Type13                           0.040528
    Soil_Type5                            0.027692
    Soil_Type36                           0.025726
    Soil_Type2                            0.022627
    Soil_Type14                           0.022019
    Elevation                             0.016090
    Soil_Type1                            0.015069
    Wilderness_Area2                      0.014994
    Soil_Type11                           0.010228
    Soil_Type16                           0.008793
    Aspect                                0.008015
    Soil_Type6                            0.006521
    Soil_Type18                           0.006312
    Soil_Type30                           0.001393
    Soil_Type34                          -0.003470
    Soil_Type8                           -0.008133
    Soil_Type25                          -0.008133
    Hillshade_9am                        -0.010286
    Horizontal_Distance_To_Hydrology     -0.010515
    Soil_Type28                          -0.012202
    Soil_Type3                           -0.016393
    Soil_Type26                          -0.017184
    Soil_Type27                          -0.023109
    Soil_Type21                          -0.024410
    Soil_Type9                           -0.027012
    Soil_Type4                           -0.027816
    Soil_Type19                          -0.031824
    Soil_Type20                          -0.053013
    Hillshade_3pm                        -0.053399
    Soil_Type33                          -0.078955
    Soil_Type31                          -0.079882
    Horizontal_Distance_To_Fire_Points   -0.089389
    Hillshade_Noon                       -0.098905
    Soil_Type24                          -0.100797
    Horizontal_Distance_To_Roadways      -0.105662
    Soil_Type12                          -0.129985
    Soil_Type32                          -0.132312
    Soil_Type23                          -0.158762
    Soil_Type22                          -0.195993
    Soil_Type29                          -0.218564
    Wilderness_Area1                     -0.230117
    Soil_Type7                                 NaN
    Soil_Type15                                NaN
    Name: Cover_Type, dtype: float64



Using `train.corr()` to calculate the correlation coefficent for all features with respect to Cover_Type shown above, we can see that some of the Soil_Type features are the most strongly correlated features.  In general, however, this correlation exercise is not particularly useful since the Cover_Type value assignment 1-7 is randomly assigned and re-ordering would change the correlation value.  The cross-correlation between features shown below using the seaborn `sns.heatmap()` tool is interesting to show the relationship between some features.  


```python
plt.subplots(figsize=(20,15))
corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 220, sep=80, n=7)

sns.heatmap(corr, mask=mask, cmap=cmap)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3e6aceb9b0>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_18_1.png)



```python
g = sns.catplot(x="Cover_Type",
            y="Elevation",
            hue="Area",
            kind="swarm",
            data=train_labels,
            height=6,
            aspect=2)
g.fig.suptitle("Wilderness Area and Elevation for each Cover Type", fontsize=20)
```




    Text(0.5,0.98,'Wilderness Area and Elevation for each Cover Type')




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_19_1.png)


Looking at the Elevation and Wilderness Area against Cover Type together starts to show some separation between Cover Types, but still there is a great deal of overlap.


```python
g = sns.catplot(x="Cover_Type",
            y="Elevation",
            hue="Soil",
            kind="swarm",
            data=train_labels,
            height=6,
            aspect=2)
g.fig.suptitle("Soil Type and Elevation for each Cover Type", fontsize=20)
```




    Text(0.5,0.98,'Soil Type and Elevation for each Cover Type')




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_21_1.png)


Plotting the Soil_Type against Elevation and Cover_Type using the Seaborn `sns.catplot(kind="swarm")` tool creates a beautiful plot and shows some separation between Soil Type and likely Cover_Type.  However there are too many Soil_Types to show good separation with colors.


```python
g = sns.catplot(x="Soil",
            y="Elevation",
            hue="Cover_Type",
            kind="swarm",
            data=train_labels,
            order=soils,
            height=6,
            aspect=2)
g.fig.suptitle("Soil Type and Elevation for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6ad652b0>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_23_1.png)



```python
def get_soil_num(row):
    return int(row['Soil'][9:])
        
soil_number = train_labels.apply(get_soil_num, axis=1)
soil_num_df = pd.DataFrame({'Soil_Type': soil_number})
train_soil_nums = pd.concat([train_labels, soil_num_df], axis=1)

g = sns.catplot(x="Soil_Type",
            y="Area",
            hue="Cover_Type",
            data=train_soil_nums, dodge=True, jitter=0.25, height=10, aspect=1.2)

g.fig.suptitle("Soil Type and Wilderness Area for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6ad4b780>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_24_1.png)


Plotting all 3 category features Cover_Type, Wilderness_Area, and Soil_Type together shows a couple combinations of Soil_Type and Wilderness_Area that only have one Cover_Type.  But, nearly all pairs have 2 or more Cover_Types and need more features to differentiate between Cover_Types.


```python
from pandas.plotting import scatter_matrix

attributes = ["Cover_Type", "Elevation", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]
scatter_matrix(train[attributes], figsize=(12,15))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6dc37438>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6afddba8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6aef0ba8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6b2006a0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6aec3828>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6aec3208>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6d8fc6a0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6adf1278>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6d9e3a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6ae047f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6d99f860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6ad7fb00>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6af6e0b8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6ae8e400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6da56208>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3e6da6ee48>]],
          dtype=object)




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_26_1.png)



```python
sns.catplot(x="Cover_Type", y="Aspect", hue="Soil", data=train_labels, jitter=0.4, height=6, aspect=2)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6dc52358>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_27_1.png)



```python
sns.catplot(x="Cover_Type", y="Horizontal_Distance_To_Roadways", hue="Soil", data=train_labels, jitter=0.4, height=6, aspect=2)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6db3c828>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_28_1.png)



```python
sns.catplot(x="Cover_Type", y="Vertical_Distance_To_Hydrology", hue="Soil", data=train_labels, jitter=0.4, height=6, aspect=2)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6db62f60>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_29_1.png)



```python
sns.catplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points", hue="Soil", data=train_labels, jitter=0.4, height=6, aspect=2)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e690f25c0>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_30_1.png)



```python
sns.catplot(x="Cover_Type", y="Slope", hue="Soil", data=train_labels, jitter=0.4, height=6, aspect=2)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e68e8d9b0>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_31_1.png)



```python
g = sns.catplot(x="Soil",
            y="Elevation",
            hue="Cover_Type",
            kind="swarm",
            data=train_labels,
            order=soils,
            height=6,
            aspect=2)
g.fig.suptitle("Soil Type and Elevation for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6b234b00>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_32_1.png)



```python
g = sns.catplot(x="Soil",y="Aspect",hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Aspect for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e6912fcc0>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_33_1.png)



```python
g = sns.catplot(x="Soil",y="Slope",hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Slope for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e68af8fd0>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_34_1.png)



```python
g = sns.catplot(x="Soil",y='Horizontal_Distance_To_Hydrology',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Horizontal_Distance_To_Hydrology for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e68af84a8>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_35_1.png)



```python
g = sns.catplot(x="Soil",y='Vertical_Distance_To_Hydrology',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Vertical_Distance_To_Hydrology for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e688e70b8>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_36_1.png)



```python
g = sns.catplot(x="Soil",y='Horizontal_Distance_To_Roadways',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Horizontal_Distance_To_Roadways for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e689a5518>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_37_1.png)



```python
g = sns.catplot(x="Soil",y='Hillshade_9am',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Hillshade_9am for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e68751b00>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_38_1.png)



```python
g = sns.catplot(x="Soil",y='Hillshade_Noon',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Hillshade_Noon for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e684e0ac8>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_39_1.png)



```python
g = sns.catplot(x="Soil",y='Hillshade_3pm',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Hillshade_3pm for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e683937b8>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_40_1.png)



```python
g = sns.catplot(x="Soil",y='Horizontal_Distance_To_Fire_Points',hue="Cover_Type",kind="swarm",data=train_labels,order=soils,height=6,aspect=2)
g.fig.suptitle("Soil Type and Horizontal_Distance_To_Fire_Points for each Cover Type", fontsize=20)
g.set_xticklabels(rotation=45)
```




    <seaborn.axisgrid.FacetGrid at 0x7f3e68cd24a8>




![png]({{site.url}}/assets/Project1_ForestCover_Part1_EDA_files/Forest%20Cover%20Type%20-%20Exploritory%20Data%20Analysis_41_1.png)


## Conclusions

No combination of Soil_Type and single Quantitative attribute draws good separation between the Cover_Type for each instance.  There does seem to be some information contained in each attribute which could help differentiate between Cover_Type, this may be enough to build a model from when all attributes are considered together, but this isn't readily obvious.

## Next Steps

Combining Decision Trees and Support Vector Machine to evaluate Engineered Features from the given attributes may help improve the ability for a model to predict Cover_Type.
