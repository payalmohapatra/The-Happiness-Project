## This file uses the regional data and does the following :
#1. Data preprocessing - Make dataframes for each regional_model
#2. Deplot LR and RF models for each
#3. Model selection and save in model_{regional_index}.pkl
import pandas as pd
import numpy as np
import seaborn as sns
from discover_feature_relationships import discover
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
import pickle
# data input
datapath = 'Dataset/'
csvname = datapath + '2015.csv'
df_2015 = pd.read_csv("Dataset/2015.csv")
df_2016 = pd.read_csv("Dataset/2016.csv")
#df_2017 = pd.read_csv("Dataset/2017.csv")

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
## Select 4 best features function

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

regional_model = []
## Manually dropping the features
for i in range (10) :
    if i == 0 :
        df_current = df_western_europe
        df_current = df_western_europe.drop(['Life','Generosity'], axis ='columns') 
    elif i == 1:    
        df_current = df_north_america
        df_current = df_north_america.drop(['GDP','Family'], axis = 'columns')
    elif i == 2:    
        df_current = df_midle_east_north_africa
        df_current = df_midle_east_north_africa.drop(['Trust','Generosity'], axis = 'columns')
    elif i == 3:    
        df_current = df_southeastern_asia
        df_current = df_southeastern_asia.drop(['Generosity','Freedom'], axis = 'columns')
    elif i == 4:    
        df_current = df_latin_america_caribbean
        df_current = df_latin_america_caribbean.drop(['Generosity','Freedom'], axis = 'columns')
    elif i == 5:    
        df_current = df_eastern_asia
        df_current = df_eastern_asia.drop(['Trust','Freedom'], axis = 'columns')
    elif i == 6:    
        df_current = df_australia
        df_current = df_australia.drop(['GDP','Trust'], axis = 'columns')
    elif i == 7:    
        df_current = df_subsaharan_africa                        
        df_current = df_subsaharan_africa.drop(['Trust','Life'], axis = 'columns')
    elif i == 8:    
        df_current = df_south_asia
        df_current = df_south_asia.drop(['Generosity','Freedom'], axis = 'columns')
    elif i == 9:    
        df_current = df_central_eastern_europe
        df_current = df_central_eastern_europe.drop(['Generosity','Life'], axis = 'columns')

    #print('Value of df is', df_current['Region'].values[0], 'for iteration ',i)
    
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    df_current.drop(['Happiness Score','Region'], axis='columns'), df_current['Happiness Score'], test_size=0.2)

    in_df = df_current.drop(columns = ['Region'])
    features = np.asarray(in_df.columns)
    select_features = SelectKBest(f_regression, k=4)
    X_new = select_features.fit_transform(X_train, Y_train)
    filter = select_features.get_support(indices = True)

    ## Selecting the top 4 features that correlate the highest with happiness factor
    df_kbest = pd.DataFrame()
    df_kbest = in_df[[str(features[filter[0]]), str(features[filter[1]]), str(features[filter[2]]), str(features[filter[3]]), 'Happiness Score']]
    #print('\n ********************Data Frame with 4 best features*********************** \n')
    #print(df_kbest)
    
    spearman_cormatrix= df_kbest.corr(method='spearman')
    #print('\n ********************Spearman Correlation*********************** \n')
    #print(spearman_cormatrix)
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

    ################## Random Forest Regression Model ########################################################
    rndm_frst_mdl = sklearn.ensemble.RandomForestRegressor()
    RF_model = rndm_frst_mdl.fit(x_training, y_training)
    y_validating_rndm_frst_p = rndm_frst_mdl.predict(x_validating)
    RF_RMSE = np.sqrt(sklearn.metrics.mean_squared_error(y_validating_rndm_frst_p, y_validating))
    print('Random Forest RMSE:', RF_RMSE)
        
    ################## Linear Regression Model ########################################################
    LR = LinearRegression()
    # fitting the training data
    LR_model = LR.fit(x_training,y_training)
    y_validating_lr_p =  LR.predict(x_validating)
    LR_RMSE = np.sqrt(sklearn.metrics.mean_squared_error(y_validating_lr_p, y_validating))
    print('LR RMSE:', LR_RMSE)

    ## Dump the model with lowest RMSE in a pickle file
    pkl_filename = "regional_model_{0}.pkl".format(i)
    if (RF_RMSE < LR_RMSE) :
        print('Choosing RF model for Region',i)
        with open(pkl_filename, 'wb') as file:
          pickle.dump(RF_model, file)
    else:
        ## Append LR to model 
        print('Choosing LR model for Region',i)
        with open(pkl_filename, 'wb') as file:
          pickle.dump(LR_model, file)   

