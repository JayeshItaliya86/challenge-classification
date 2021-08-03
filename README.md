# challenge-classification
Challenge classification
Contributors:
Jacques Declercq
Jayesh Italiya
Graciela Lopez Rosson
Bilal Mesmoudi
Amaury van Kesteren

# 0. Setting up the project

Conda packages
Github repo link: https://github.com/JayeshItaliya86/challenge-classification
ReadMe file
Importing:

```
# Import libraries

#Data analysis libraries
import numpy as np 
import pandas as pd 

#Visulization and statistics libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from scipy import fftpack
import seaborn as sns
style.use('seaborn')

# Model preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

# Medel related libraries
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

```

# 1. data gathering and exploration
Original dataset is composed of 2 csv files, one training data and one test data.
The test data informs us that 112 experiments were done, 12 turned out to result in a correct bearings and 100 in faulty bearings.
The training data has one target variable and 13 potential Xs:
- Watts and RPM and Hz columns hint at speed engine
- Experiment ID, Bearing ID 1 and Bearing ID 2 give further contextual information on the experiment
- 6 columns reveal vibrations measured by the 2 accelerometers on the two bearings. The vibrations are measured on 3 axes: x, y, z:
        - One bearing is systematicaly the same in each experiment
        - The second bearing is the core of the experiment. The variations on the 3 axes giving the raw data on what type and amount of vibrations results in a correct or faulty bearing.
- Another interesting column is the timestamp column.

Initial plotting for data exploration:

![spectrum_frequence](https://user-images.githubusercontent.com/84380197/128018272-96c708e6-9d28-4f97-88b4-255ed3c648d6.png)

![spectrum_frequence2](https://user-images.githubusercontent.com/84380197/128019487-13b648c4-caaa-43e1-916f-418449ec3873.png)

![x axis superposition](https://user-images.githubusercontent.com/84380197/128018636-4b258c3a-abd6-4696-85df-d73a023fcda2.png)

![z axis](https://user-images.githubusercontent.com/84380197/128018436-3363fa2f-53b6-4c77-889a-d201c591603c.png)

![x axis](https://user-images.githubusercontent.com/84380197/128018444-cd1618b3-f469-4597-b24d-ee50b5d4a50a.png)

![y index](https://user-images.githubusercontent.com/84380197/128018445-d978bcb9-5e57-4a0b-803c-958884a13747.png)

![heatmap](https://user-images.githubusercontent.com/84380197/128018543-12c4c051-0056-4036-b6d0-74da285e9d9a.png)

# 2. data preprocessing

The following columns are dropped:
- experiment_id
- bearing_1_id
- timestamp

```
df_train = origin_set.drop(['experiment_id','bearing_1_id', 'timestamp'], axis=1)
```

We rename the columns.
```
# To rename the column name using function
"""This function takes dataframe & prefix of the columns.
   It needs name of columns from the dataframe and add prefix before the each name of the columns.
   It returns dataframe with new column names."""
def rename_column(df,prefix):
    column_name = list(df.columns)
    column_name = [prefix + name for name in column_name]
    return df.set_axis(column_name, axis=1)
```

## Statistical feature engineering on the dimensions of the two bearings

Vibrations on the first bearing and the second bearing are reworked to obtain the following derived features:
- max
- min
- mean
- standard deviation
- median
- range (difference max-min)
- kurtosis
- skewness


This results in 24 additional Xs.

```
max_set = df_train.groupby(['bearing_2_id']).max()
max_set = rename_column(max_set,"max_")
min_set = df_train.groupby(['bearing_2_id']).min()
min_set = rename_column(min_set,"min_")
mean_set = df_train.groupby(['bearing_2_id']).mean()
mean_set = rename_column(mean_set,"mean_")
std_set = df_train.groupby(['bearing_2_id']).std()
std_set = rename_column(std_set,"std_")
median_set = df_train.groupby(['bearing_2_id']).median()
median_set = rename_column(median_set,"median_")
range_set = df_train.groupby(['bearing_2_id']).max() - df_train.groupby(['bearing_2_id']).min()
range_set = rename_column(range_set,"range_")
kurtosis_set = df_train.groupby(['bearing_2_id']).apply(pd.DataFrame.kurtosis)
kurtosis_set = rename_column(kurtosis_set, 'kurtosis_')
skew_set = df_train.groupby(['bearing_2_id']).skew()
skew_set = rename_column(skew_set, 'skew_')
```
To make dataframe of individual features and make list of new dataframe and list of new column names
```
df_bearing_1_x = df_train.drop(['a2_x','a2_y','a2_z','a1_y','a1_z'], axis = 1)
df_bearing_1_y = df_train.drop(['a2_x','a2_y','a2_z','a1_x','a1_z'], axis = 1)
df_bearing_1_z = df_train.drop(['a2_x','a2_y','a2_z','a1_x','a1_y'], axis = 1)
df_bearing_2_x = df_train.drop(['a1_x','a1_y','a1_z','a2_y','a2_z'], axis = 1)
df_bearing_2_y = df_train.drop(['a1_x','a1_y','a1_z','a2_x','a2_z'], axis = 1)
df_bearing_2_z = df_train.drop(['a1_x','a1_y','a1_z','a2_x','a2_y'], axis = 1)
bearing_feature = [df_bearing_1_x,df_bearing_1_y,df_bearing_1_z,df_bearing_2_x,df_bearing_2_y,df_bearing_2_z]
list_column = ['fft_a1_x','fft_a1_y','fft_a1_z','fft_a2_x','fft_a2_y','fft_a2_z']
```

## Fast Fourier Transformation on the 3 axes of bearing one and two column

```
def by_axis_bearing(bearing_feature,i):
    max_list = []
    for index in range(len(bearing_feature)):
        bearing_idx = bearing_feature[index]
        df_bearing = bearing_idx[bearing_idx['bearing_2_id'] == i]
        fft_values = fftpack.fft(df_bearing)
        max_amplitude = np.argmax(np.abs(fft_values))
        max_list.append(max_amplitude)
    return max_list

#def by_bearing(df):

number_bearing = df_train['bearing_2_id'].max()

max_list = by_axis_bearing(bearing_feature,1)
new_set = pd.DataFrame([max_list],columns=list_column,index=[1])

for i in range(2,number_bearing+1):
    max_list = by_axis_bearing(bearing_feature,i)
    temp_set = pd.DataFrame([max_list],columns=list_column)
    new_set = new_set.append(temp_set, ignore_index=True)
```
![fft](https://user-images.githubusercontent.com/84380197/128018059-eb45a8c5-69ea-4c84-81f1-a41c85dbee25.png)

![fft2](https://user-images.githubusercontent.com/84380197/128019714-37dba357-028f-4406-be97-7eaabfdfd2de.png)

![fft110](https://user-images.githubusercontent.com/84380197/128024947-79f5314b-7911-4db5-ac1f-9456dcfb8713.png)

![fft1](https://user-images.githubusercontent.com/84380197/128024963-5d778d03-0cc4-4a3c-90e0-828e5961c91d.png)

## Merge all the dataframes
```
df = pd.concat([max_set, min_set, mean_set, std_set, median_set, range_set, kurtosis_set, skew_set, new_set], axis=1)
df['target']= target_set.iloc[1:,1]
```

This results in an amplitude variable per experiment per axe or in 6 additonal features.

We obtain a grand total of 80 features.

# 3. Choosing, training and scoring a model

Several models are tried, trained, grid searched and scored.
Choosing the model is done by looking at the score, including cross-validation score, classification report, and confusion matrix.
Random state is set at 0.41 and test size at 0.2.
Grid search parameters:
- n_estimators:[200,150,100,50],
- criterion:['gini','entropy'],
- max_depth:[2,4,6,8]

```

X = df.drop(['kurtosis_bearing_2_id', 'target'], axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

param_grid = {
    'n_estimators':[200,150,100,50],
    'criterion':['gini','entropy'],
    'max_depth':[2,4,6,8]
}
```

## 3.1 Random forest 

```

Model_rf = RandomForestClassifier()
grid = GridSearchCV(estimator=Model_rf, param_grid=param_grid, n_jobs=-1,cv=5)
grid.fit(X_train, y_train)
Model_rf = grid.best_estimator_
y_pred_train = Model_rf.predict(X_test)
training_data_accuracy_rf = accuracy_score(y_test, y_pred_train)
print("The accuracy of RandomForestC Model is", (training_data_accuracy_rf*100), '%')
print('The Cross Validation Rapport : ','\n', (classification_report(y_test, y_pred_train)))
confusion_matrix(y_test,y_pred_train)

```

## 3.2 Linear Regression

```
reg = LinearRegression()
cv_scores = cross_val_score(reg, X, y, cv = 5)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
```

## 3.3 SVM

```
svc = SVC(probability = True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
cv = cross_val_score(svc,X_train,y_train,cv=15)
print(cv)
print(cv.mean())
print('confusion matrix')
print(confusion_matrix(y_test, y_pred))
```

## 3.4 OLS

```
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)
predictions
```


## 3.5 KNN

```
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

# 4. Feature importance order

```
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(Model_rf.feature_importances_,3)})
importances1 = importances.sort_values('importance',ascending=False).set_index('feature')
importances1.plot(kind='bar',figsize=(15,6))
```

![importance](https://user-images.githubusercontent.com/84380197/128018633-a32c749e-adf6-49b4-bec9-8b234a24b6d1.png)
# Conclusion

The model that works best with our data is RandomForest. 
We obtain a score of 91%. 
However, our model is able to predict only 2 out of 4 good bearings. 
19 out of faulty 19 bearings were correctly predicted.

The most important feature emerging from the feature engineering is the mean.



