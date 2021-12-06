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
df_2017 = pd.read_csv("Dataset/2017_test.csv")

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
df = prep_frame(df_2017, 2017)



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

test_RMSE_hist = []
test_RMSE_avg = 0

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
    
    X_test = df_current.drop(['Happiness Score', 'Region'], axis='columns')
    Y_test = df_current['Happiness Score']
    pkl_filename = "regional_model_{0}.pkl".format(i)
    print('Reading file', pkl_filename)
    ## Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file) 
    # Calculate the accuracy score and predict target values
    Y_predict =  pickle_model.predict(X_test)
    test_RMSE = np.sqrt(sklearn.metrics.mean_squared_error(Y_predict,Y_test))
    print('Test RMSE at iteration', i ,' is', test_RMSE)
    test_RMSE_hist.append(test_RMSE)

    test_RMSE_avg += test_RMSE

    
print('Average Test RMSE from Ensemble learning is ', test_RMSE_avg/10)     