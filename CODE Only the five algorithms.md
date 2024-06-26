```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
sns.set_style("darkgrid")
tfont = {'fontsize':15, 'fontweight':'bold'}
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn import set_config
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score,StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn .metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics 
from sklearn.metrics import precision_recall_fscore_support as score,roc_auc_score
from sklearn.metrics import auc
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import auc
import numpy as np
from sklearn.utils import resample
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import NuSVC
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
```


```python
data=pd.read_excel(r'C:\Users\path.xlsx')
```


```python
X = data.drop(columns='reponse_ligne_un')
y = data['reponse_ligne_un']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 1)
```


```python
cat_col = list(X_train.select_dtypes('object'))
float_col = list(X_train.select_dtypes(float))

cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                         OneHotEncoder(handle_unknown='ignore'))

float_pipe = make_pipeline(KNNImputer())

preprocessor = make_column_transformer((cat_pipe, cat_col),
                                       (float_pipe, float_col),
                                       remainder='passthrough')
```


```python
# foret aléatoire - Random forest
```


```python
foret_aleatoire = make_pipeline(preprocessor,RandomForestClassifier())

params = {'randomforestclassifier__n_estimators':[50,75],
          'randomforestclassifier__max_depth': [3,5],
          'randomforestclassifier__max_samples':[0.5, 0.75],
          'randomforestclassifier__min_samples_split': [2, 5, 10],
          'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        'randomforestclassifier__max_features': ['sqrt', 'log2', None]
         }
                            
grid = GridSearchCV(foret_aleatoire, param_grid=params, verbose=1, scoring='recall',cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
foretaleatoire = grid.best_estimator_
```


```python
predictions = foretaleatoire.predict(X_test)
```


```python
# XGBoost 
```


```python
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')

# Définir les hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='recall', cv=10, verbose=1)

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

y_pred = grid_search.best_estimator_.predict(X_test)
```


```python
# SVM 
```


```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```


```python
X_test_scaled = scaler.transform(X_test)
```


```python
svc = NuSVC()

# Définir la grille des paramètres à rechercher
param_grid = {
    
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=10, verbose=2, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters found: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred_deux = best_model.predict(X_test_scaled)
```


```python
# CNB
```


```python
cat_nb = CategoricalNB()
```


```python
cat_nb.fit(X_train, y_train)
```


```python
y_pred_2 = cat_nb.predict(X_test)
```


```python
# Logistic regression 
```


```python
log_regression_alg = LogisticRegression(solver='liblinear', max_iter=30)
```


```python
pipeline_trois = Pipeline([
                          ('stand', StandardScaler()),
                          ('logreg',log_regression_alg)
])
```


```python
pipeline_trois.fit(X_train,y_train)
```


```python
y_pred_trois = pipeline_trois.predict(X_test)
```


```python
# End
```


```python

```
