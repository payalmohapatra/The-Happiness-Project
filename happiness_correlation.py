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
df_2017 = pd.read_csv("Dataset/2017.csv")
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
df = df.append(prep_frame(df_2017, 2017), sort=False)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
   df.drop(['Happiness Score'], axis='columns'), df['Happiness Score'], test_size=0.2)

##############################################################################################################
##############################################################################################################
## Use kbest features function for regression to get 4 best features for the whole data
features = np.asarray(df.columns)
select_features = SelectKBest(f_regression, k=4)
X_new = select_features.fit_transform(X_train, Y_train)
filter = select_features.get_support(indices = True)
#print("Best Selected features:")
#print(features[filter[0]])
#print(features[filter[1]])
#print(features[filter[2]])
#print(features[filter[3]])
#print('\n')

df_kbest = pd.DataFrame()
df_kbest = df[[str(features[filter[0]]), str(features[filter[1]]), str(features[filter[2]]), str(features[filter[3]]), 'Happiness Score']]
#print('\n ********************Data Frame with 4 best features*********************** \n')
#print(df_kbest)

spearman_cormatrix= df_kbest.corr(method='spearman')
#print('\n ********************Spearman Correlation*********************** \n')
#print(spearman_cormatrix)

#plt.figure(1)
#sns.heatmap(spearman_cormatrix, vmin=-1, vmax=1, center=0, cmap="viridis", annot=True)
#plt.show()

## TODO:: Automate this selection ##########
### Drop Life since GDP and Life highly correlate (0.8)
df_kbest_corr = pd.DataFrame()
df_kbest_corr = df_kbest.drop(['Life'], axis='columns')
#print('\n ********************Data Frame with 3 best features*********************** \n')
#print(df_kbest_corr)

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

#print('Size of x Train', np.shape(x_training))
#print('Size of x Validate', np.shape(x_validating))
#print('Size of x Test', np.shape(x_testing))
#
#print('Size of y Train', np.shape(y_training))
#print('Size of y Validate', np.shape(y_validating))
#print('Size of y Test', np.shape(y_testing))
################## Random Forest Regression Model ########################################################
rndm_frst_mdl = sklearn.ensemble.RandomForestRegressor()
rndm_frst_mdl.fit(x_training, y_training)
y_validating_rndm_frst_p = rndm_frst_mdl.predict(x_validating)
print('Random Forest RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_validating_rndm_frst_p, y_validating)))

#rndm_frst_mdl_full = sklearn.ensemble.RandomForestRegressor()
#rndm_frst_mdl_full.fit(X_train, Y_train)
#y_validating_p = rndm_frst_mdl_full.predict(X_test)
#print('Random Forest RMSE without feature reduction:', np.sqrt(sklearn.metrics.mean_squared_error(y_validating_p, Y_test)))
################## Linear Regression Model ########################################################
LR = LinearRegression()
# fitting the training data
LR.fit(x_training,y_training)
y_validating_lr_p =  LR.predict(x_validating)
print('LR RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_validating_lr_p, y_validating)))
