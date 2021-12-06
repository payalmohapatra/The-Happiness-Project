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

## Pre-process data :: Append the years and generate a complete dataset
h_cols = ['Country', 'Region', 'GDP', 'Family', 'Life', 'Freedom', 'Generosity', 'Trust']
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

unique_continents = df['Region'].nunique(dropna=False) 

##############################################################################################################
## Extracting raws with regional columns
df_western_europe = df[df['Region'] == 'Western Europe']

df_north_america = df[df['Region'] == 'North America']

df_australia = df[df['Region'] == 'Australia and New Zealand']

df_central_eastern_europe = df[df['Region'] == 'Central and Eastern Europe']

df_latin_america_caribbean = df[df['Region'] == 'Latin America and Caribbean']

df_eastern_asia = df[df['Region'] == 'Eastern Asia']

df_southeastern_asia = df[df['Region'] == 'Southeastern Asia']

df_midle_east_north_africa = df[df['Region'] == 'Middle East and Northern Africa']

df_subsaharan_africa = df[df['Region'] == 'Sub-Saharan Africa']

df_south_asia = df[df['Region'] == 'Southern Asia']


for i in range (10) :
    if i == 0 :
        df_current = df_western_europe
    elif i == 1:    
        df_current = df_north_america
    elif i == 2:    
        df_current = df_midle_east_north_africa
    elif i == 3:    
        df_current = df_southeastern_asia
    elif i == 4:    
        df_current = df_latin_america_caribbean
    elif i == 5:    
        df_current = df_eastern_asia
    elif i == 6:    
        df_current = df_australia
    elif i == 7:    
        df_current = df_subsaharan_africa                        
    elif i == 8:    
        df_current = df_south_asia
    elif i == 9:    
        df_current = df_central_eastern_europe

    print('Value of df is', df_current['Region'].values[0], 'for iteration ',i)
    
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    df_current.drop(['Happiness Score','Region'], axis='columns'), df_current['Happiness Score'], test_size=0.2)

    in_df = df_current.drop(columns = ['Region'])
    features = np.asarray(in_df.columns)
    select_features = SelectKBest(f_regression, k=4)
    X_new = select_features.fit_transform(X_train, Y_train)
    filter = select_features.get_support(indices = True)
    print("Best Selected features:")
    print(features[filter[0]])
    print(features[filter[1]])
    print(features[filter[2]])
    print(features[filter[3]])
    print('\n')

    ## Selecting the top 4 features that correlate the highest with happiness factor
    df_kbest = pd.DataFrame()
    df_kbest = in_df[[str(features[filter[0]]), str(features[filter[1]]), str(features[filter[2]]), str(features[filter[3]]), 'Happiness Score']]
    #print('\n ********************Data Frame with 4 best features*********************** \n')
    #print(df_kbest)
    
    spearman_cormatrix= df_kbest.corr(method='spearman')
    print('\n ********************Spearman Correlation*********************** \n')
    print(spearman_cormatrix)
    #df_current = select_kfeatures(df_western_europe)
    
    ## Select the factors that have the highest cross-correlation

    plt.figure(i)
    sns.heatmap(spearman_cormatrix, vmin=-1, vmax=1, center=0, cmap="viridis", annot=True)

    train, validate, test = \
                  np.split(df_kbest.sample(frac=1, random_state=42), 
                           [int(.7*len(df_kbest)), int(.85*len(df_kbest))])
        
    x_training   = train.drop(['Happiness Score'], axis='columns')
    x_validating = validate.drop(['Happiness Score'], axis='columns')
    x_testing    = test.drop(['Happiness Score'], axis='columns')
    y_training   = train['Happiness Score']
    y_validating = validate['Happiness Score']
    y_testing    = test['Happiness Score']

    print('Number of samples for ',df_current['Region'].values[0], ' are ',df_kbest.shape[0])
    print('Training set size ',x_training.shape[0])
    print('Validating set size ',x_validating.shape[0])
    print('Testing set size ',x_testing.shape[0])
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


plt.show()