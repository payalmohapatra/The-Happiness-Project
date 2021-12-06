import pandas as pd
import numpy as np
import seaborn as sns
from discover_feature_relationships import discover
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
# data input
datapath = 'Dataset/'
csvname = datapath + '2015.csv'
df_2015 = pd.read_csv("Dataset/2015.csv")
df_2016 = pd.read_csv("Dataset/2016.csv")
##############################################################################################################
##############################################################################################################
# Pre-process data :: Append the years and generate a complete dataset
h_cols = ['Country', 'GDP', 'Family', 'Life', 'Freedom', 'Generosity', 'Trust']
def prep_frame(df_year, year):
    df = pd.DataFrame()
    # Work around to load 2015, 2016, 2017 data into one common column
    target_cols = []
    for c in h_cols:
        target_cols.extend([x for x in df_year.columns if c in x])
    df[h_cols] = df_year[target_cols]
    df['Happiness Score'] = df_year[[x for x in df_year.columns if 'Score' in x]]
    # Append year and assign to multi-index
    df['Year'] = year
    df = df.set_index(['Country', 'Year'])
    return df
df = prep_frame(df_2015, 2015)
df = df.append(prep_frame(df_2016, 2016), sort=False)
#df = df.append(prep_frame(df_2017, 2017), sort=False)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
   df.drop(['Happiness Score'], axis='columns'), df['Happiness Score'], test_size=0.2)

##############################################################################################################
##############################################################################################################
## Use kbest features function for regression to get 4 best features for the whole data
features = np.asarray(df.columns)
select_features = SelectKBest(f_regression, k=4)
X_new = select_features.fit_transform(X_train, Y_train)
filter = select_features.get_support(indices = True)
print("Best Selected features:")
print(features[filter[0]])
print(features[filter[1]])
print(features[filter[2]])
print(features[filter[3]])
print('\n')

df_kbest = pd.DataFrame()
df_kbest = df[[str(features[filter[0]]), str(features[filter[1]]), str(features[filter[2]]), str(features[filter[3]]), 'Happiness Score']]
print('\n ********************Data Frame with 4 best features*********************** \n')
print(df_kbest)

spearman_cormatrix= df_kbest.corr(method='spearman')
print('\n ********************Spearman Correlation*********************** \n')
print(spearman_cormatrix)

plt.figure(1)
sns.heatmap(spearman_cormatrix, vmin=-1, vmax=1, center=0, cmap="viridis", annot=True)
plt.show()

### Drop Life since GDP and Life highly correlate (0.8)

df_kbest_corr = pd.DataFrame()
df_kbest_corr = df_kbest.drop(['Life'], axis='columns')
print('\n ********************Data Frame with 3 best features*********************** \n')
print(df_kbest_corr)

##############################################################################################################
##############################################################################################################

## Shuffle data and divide into train, validate and Test ##
train, validate, test = \
              np.split(df_kbest_corr.sample(frac=1, random_state=42), 
                       [int(.8*len(df_kbest_corr)), int(.9*len(df_kbest_corr))])

x_training   = train.drop(['Happiness Score'], axis='columns')
x_validating = validate.drop(['Happiness Score'], axis='columns')
x_testing    = test.drop(['Happiness Score'], axis='columns')
y_training   = train['Happiness Score']
y_validating = validate['Happiness Score']
y_testing    = test['Happiness Score']
################## Random Forest Regression Model ########################################################
rndm_frst_mdl = sklearn.ensemble.RandomForestRegressor()
rndm_frst_mdl.fit(x_training, y_training)
y_validating_rndm_frst_p = rndm_frst_mdl.predict(x_validating)
print('Random Forest RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_validating_rndm_frst_p, y_validating)))

################## Linear Regression Model ########################################################
LR = LinearRegression()
# fitting the training data
LR.fit(x_training,y_training)
y_validating_lr_p =  LR.predict(x_validating)
print('LR RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_validating_lr_p, y_validating)))

plt.figure(1)
plt.scatter(y_validating_lr_p, y_validating)
plt.title('Scatter plot between LR models')
plt.figure(2)
plt.scatter(y_validating_rndm_frst_p, y_validating)
plt.title('Scatter plot between random forest model')
#plt.show()


###############################################################################################################
###############################################################################################################
## Train with full dataset features
train_full, validate_full, test_full = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.9*len(df)), int(.95*len(df))])

x_full_training   = train_full.drop(['Happiness Score'], axis='columns')
x_full_validating = validate_full.drop(['Happiness Score'], axis='columns')
x_full_testing    = test_full.drop(['Happiness Score'], axis='columns')
y_full_training   = train_full['Happiness Score']
y_full_validating = validate_full['Happiness Score']
y_full_testing    = test_full['Happiness Score']
################### Random Forest Regression Model ########################################################
rndm_frst_mdl_full = sklearn.ensemble.RandomForestRegressor()
rndm_frst_mdl_full.fit(x_full_training, y_full_training)
y_full_validating_rndm_frst_p = rndm_frst_mdl_full.predict(x_full_validating)
print('Random Forest Full RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_full_validating_rndm_frst_p, y_full_validating)))

################### Linear Regression Model ########################################################
LR_full = LinearRegression()
# fitting the training data
LR_full.fit(x_full_training,y_full_training)
y_full_validating_lr_p =  LR_full.predict(x_full_validating)
print('LR Full RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_full_validating_lr_p, y_full_validating)))

plt.figure(3)
plt.scatter(y_full_validating_lr_p, y_full_validating)
plt.title('Scatter plot between LR full models')
plt.figure(4)
plt.scatter(y_full_validating_rndm_frst_p, y_full_validating)
plt.title('Scatter plot between random forest full model')
#plt.show()


#########################################################################################################
## Use the developed model to do inference in 2017 dataset
df_2017 = pd.read_csv("Dataset/2017_test.csv")
# Pre-process data :: Append the years and generate a complete dataset
h_cols = ['Country', 'GDP', 'Family', 'Life', 'Freedom', 'Generosity', 'Trust']

df_test = prep_frame(df_2017, 2017)
print(df_test)
X_test_full = df_test.drop(['Happiness Score'], axis='columns')
Y_test_full = df_test['Happiness Score']

### Drop the columns that were least correlating with happiness score and redundan
### Keep :: GDP | Family | Freedom

df_test_rdcd_ftr = df_test.drop(['Life', 'Generosity', 'Trust'], axis='columns')
X_test_rdcd_ftr = df_test_rdcd_ftr.drop(['Happiness Score'], axis='columns')
Y_test_rdcd_ftr = df_test_rdcd_ftr['Happiness Score']
### Model I     :: Full features with Linear Regression model #####
Y_test_full_lr_p =  LR_full.predict(X_test_full)
print('Test :: LR Full RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(Y_test_full_lr_p, Y_test_full)))
### Model II    :: Full features with Random Forest model #####
Y_test_full_rf_p =  rndm_frst_mdl_full.predict(X_test_full)
print('Test :: random forest Full RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(Y_test_full_rf_p, Y_test_full)))
### Model III   :: reduced features with Linear Regression model #####
Y_test_rdcd_ftr_lr_p =  LR.predict(X_test_rdcd_ftr)
print('Test :: LR Reduced Features fit RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(Y_test_rdcd_ftr_lr_p, Y_test_rdcd_ftr)))
### Model IV    :: reduced features with Random Forest model #####
Y_test_rf_p =  rndm_frst_mdl.predict(X_test_rdcd_ftr)
print('Test :: random forest Reduced Features fit RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(Y_test_rf_p, Y_test_rdcd_ftr)))