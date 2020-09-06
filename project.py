import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('housing_train2.csv')
#Exploratory Data Analysis 
df.shape
df['Type'].unique()
df.describe()
df.info()
df_category['Method'].value_counts()
df.isnull().values.sum()
df.isnull().sum()
df.isnull().mean()*100
df.nunique()
correlation = df.corr()

sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
sns.pairplot(df)
sns.distplot(df['Landsize'])

#visualizaing missing values in categorical feature "Method"
method_count = df['Method'].value_counts()
sns.set(style="darkgrid")
sns.barplot(method_count.index, method_count.values, alpha=0.9)
plt.title('Frequency Distribution of Method')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Method', fontsize=12)
plt.show()

##Missing value imputation for numeric variables
#Data type conversion
df.dtypes 
df['Postcode'] = df['Postcode'].astype(object) 

#Create a seperate dataframe for numeric & categorical features
df_numeric  = df.select_dtypes(exclude='object')
df_category = df.select_dtypes('object')

df_numeric = pd.DataFrame(data=df_numeric)
df_category = pd.DataFrame(data=df_category)

#MICE imputation on numerical dataset
df_numeric_MICE = df_numeric.copy(deep=True)
MICE_numeric_imputer = IterativeImputer()
df_numeric_MICE.iloc[:, :] = MICE_numeric_imputer.fit_transform(df_numeric)

#KNN imputation on numerical dataset
df_numeric_KNN = df_numeric.copy(deep=True)
KNN_numeric_imputer = KNNImputer()
df_numeric_KNN.iloc[:, :] = KNN_numeric_imputer.fit_transform(df_numeric)

#Mean imputation on numerical datasetp
df_numeric_mean = df_numeric.copy(deep=True)
mean_numeric_imputer = SimpleImputer(strategy='mean')
df_numeric_mean.iloc[:, :] = mean_numeric_imputer.fit_transform(df_numeric)

#Median imputation on numerical dataset
df_numeric_median = df_numeric.copy(deep=True)
median_numeric_imputer = SimpleImputer(strategy='median')
df_numeric_median.iloc[:, :] = median_numeric_imputer.fit_transform(df_numeric)

#Mode imputation on numerical dataset
df_numeric_mode = df_numeric.copy(deep=True)
mode_numeric_imputer = SimpleImputer(strategy='most_frequent')
df_numeric_mode.iloc[:, :] = mode_numeric_imputer.fit_transform(df_numeric)

#Evaluation of imputation meathods
#Fit linear model on all the methods

#Complete data 
df_numeric_cc = df_numeric.dropna(how='any')
XX_cc = sm.add_constant(df_numeric_cc.iloc[:, :-1])
yy_cc = df_numeric_cc['Price']
lm_cc = sm.OLS(yy_cc, XX_cc).fit()
lm_cc.summary()
#MICE
XX_MICE = sm.add_constant(df_numeric_MICE.iloc[:, :-1])
yy_MICE = df_numeric_MICE['Price']
lm_MICE = sm.OLS(yy_MICE, XX_MICE).fit()
#KNN
XX_KNN = sm.add_constant(df_numeric_KNN.iloc[:, :-1])
yy_KNN = df_numeric_KNN['Price']
lm_KNN = sm.OLS(yy_KNN, XX_KNN).fit()
#Mean 
XX_mean = sm.add_constant(df_numeric_mean.iloc[:, :-1])
yy_mean = df_numeric_mean['Price']
lm_mean = sm.OLS(yy_mean, XX_mean).fit()
#Median 
XX_median = sm.add_constant(df_numeric_median.iloc[:, :-1])
yy_median = df_numeric_median['Price']
lm_median = sm.OLS(yy_median, XX_median).fit()
#Mode
XX_mode = sm.add_constant(df_numeric_mode.iloc[:, :-1])
yy_mode = df_numeric_mode['Price']
lm_mode = sm.OLS(yy_mode, XX_mode).fit()

print(pd.DataFrame({'Complete': lm_cc.rsquared_adj,
'Mean Imp.': lm_mean.rsquared_adj,
'Median Imp.': lm_median.rsquared_adj,
'Mode Imp.': lm_mode.rsquared_adj,
'KNN Imp.': lm_KNN.rsquared_adj,
'MICE Imp.': lm_MICE.rsquared_adj},
index=['R_squared_adj']))

df_numeric_cc['Car'].plot(kind='kde', c='red', linewidth=3)
df_numeric_mean['Car'].plot(kind='kde')
df_numeric_median['Car'].plot(kind='kde')
df_numeric_mode['Car'].plot(kind='kde')
df_numeric_KNN['Car'].plot(kind='kde')
df_numeric_MICE['Car'].plot(kind='kde')
labels = ['Baseline (Complete Case)','Mean Imputation','Median Imputation','Mode Imputation','KNN Imputation','MICE Imputation']
plt.legend(labels)
plt.xlabel('Car')

#Imputation for categorical Variable
df_category.dtypes 
df_category['Method'].value_counts()
df_category2 = df_category.copy(deep=True)
OrdinalEncoder_df_category = OrdinalEncoder()
df_category2.iloc[:, :] = OrdinalEncoder_df_category.fit_transform(df_category2)

#Merging of numeric & categorical dataframe
df_main = pd.concat([df_category2, df_numeric_KNN], axis=1)
X_Corr = Corr_data.iloc[:, :-1].values
y_corr = Corr_data.iloc[:, -1].values
#Feature Selection methods
#Filter method
#Correlation 

Corr_df = df_main.copy()

Corr_selection = Corr_df.drop('Price', axis=1).apply(lambda x: x.corr(Corr_df.Price))
indices = np.argsort(Corr_selection)
Corr_selection[indices]

names=['Type','Suburb','CouncilArea','Distance','Method','SellerG', 
       'Landsize', 'Postcode', 'Car','Bathroom', 'Rooms', 'BuildingArea', 'Bedroom2']
plt.title('Features')
plt.barh(range(len(indices)), Corr_selection[indices], color='g', align='center')
plt.yticks(range(len(indices)), [names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

for i in range(0, len(indices)):
    if np.abs(Corr_selection[i])>0.4:
        print(names[i])

#Selecting the features with correlation > 0.4
Corr_data = Corr_df[['Suburb','Landsize','Car','Bathroom','Bedroom2']]        

for i in range(0,len(Corr_data.columns)):
    for j in  range(0,len(Corr_data.columns)):
        if i!=j:
            corr_1=np.abs(Corr_data[Corr_data.columns[i]].corr(Corr_data[Corr_data.columns[j]]))
            if corr_1 <0.3:
                print( Corr_data.columns[i] , " is not correlated  with ", Corr_data.columns[j])
            elif corr_1>0.75:
                print( Corr_data.columns[i] , " is highly  correlated  with ", Corr_data.columns[j])        

#Wrapper methods 
#Forward stepwise Selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

fs_df = df_main.copy()
fs_X = fs_df.drop('Price', 1)       
fs_y = fs_df['Price']  

fs = SFS(LinearRegression(),
           k_features=4,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)
fs.fit(fs_X, fs_y)
fs.k_feature_names_

#Backward Elimination
be_df = df_main.copy()
be_X = be_df.drop('Price', 1)       
be_y = be_df['Price']  

be = SFS(LinearRegression(), 
          k_features=4, 
          forward=False, 
          floating=False,
          cv=0)
be.fit(be_X, be_y)
be.k_feature_names_

#Finding optimal number of features for forward & Backward by hit and trial method
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fs1 = SFS(LinearRegression(), 
          k_features=(3,11), 
          forward=True, 
          floating=False,
          cv=0)
fs1.fit(fs_X, fs_y)

fig = plot_sfs(fs1.get_metric_dict(), kind='std_dev')
plt.title('Forward Selection (w. StdErr)')
plt.grid()
plt.show()

be1 = SFS(LinearRegression(), 
          k_features=(3,11), 
          forward=False, 
          floating=False,
          cv=0)
be1.fit(be_X, be_y)

fig = plot_sfs(be1.get_metric_dict(), kind='std_dev')
plt.title('Backward Elimination (w. StdErr)')
plt.grid()
plt.show()

#Bi-directional elimination(the best subset selection)
bi_df = df_main.copy()
bi_X = bi_df.drop('Price', 1)       
bi_y = bi_df['Price'] 
bi = SFS(LinearRegression(), 
          k_features=4, 
          forward=True, 
          floating=True,
          cv=0)
bi.fit(bi_X, bi_y)
bi.k_feature_names_

#Embedded methods
#Ridge Regreesion
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#Feature Scaling methods
#Standardization
sc = StandardScaler()
df_standardscaler = df_main.copy()
df_standardscaler = sc.fit_transform(df_standardscaler)

#Min-Max Scaler
mm = MinMaxScaler()
df_MinMaxScaler = df_main.copy()
df_MinMaxScaler = mm.fit_transform(df_MinMaxScaler)

#MaxAbsScaler
mbs = MaxAbsScaler()
df_MaxAbsScaler = df_main.copy()
df_MaxAbsScaler= mbs.fit_transform(df_MaxAbsScaler)

#RobustScaler
RS = RobustScaler()
df_RobustScaler = df_main.copy()
df_RobustScaler = RS.fit_transform(df_RobustScaler)

#Quantile Transformer
QT = QuantileTransformer()
df_QuantileTransformer = df_main.copy()
df_QuantileTransformer = QT.fit_transform(df_QuantileTransformer)

#Spliting the data into training and validation
df_standardscaler = pd.DataFrame(data=df_standardscaler)
X = df_standardscaler.iloc[:, :-1].values
y = df_standardscaler.iloc[:, -1].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_train_lr= lr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr= lr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 
print(r2_score(y_test, pred_test_lr))

#Ridge regression
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))   

#lasso regression
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))

#ElasticNet Regression
model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(X_train, y_train) 
pred_train_enet= model_enet.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_enet)))
print(r2_score(y_train, pred_train_enet))

pred_test_enet= model_enet.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_enet)))
print(r2_score(y_test, pred_test_enet))

#SVR
model_svr = SVR(kernel = 'rbf')
model_svr.fit(X_train, y_train)
predict_test_svr = model_svr.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,predict_test_svr)))
print(r2_score(y_test, predict_test_svr))






















