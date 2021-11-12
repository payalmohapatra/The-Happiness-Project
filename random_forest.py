import pandas as pd
import numpy as np
import seaborn as sns
from discover_feature_relationships import discover
import sklearn
import matplotlib.pyplot as plt

# data input
datapath = 'Dataset/'
csvname = datapath + '2015.csv'
#data = np.loadtxt(csvname,delimiter = ',')
df_2015 = pd.read_csv("Dataset/2015.csv")
df_2016 = pd.read_csv("Dataset/2016.csv")
df_2017 = pd.read_csv("Dataset/2017.csv")

targets = ['Low', 'Low-Mid', 'Top-Mid', 'Top']
h_cols = ['Country', 'GDP', 'Family', 'Life', 'Freedom', 'Generosity', 'Trust']
def prep_frame(df_year, year):
    df = pd.DataFrame()
    # Work around to load 2015, 2016, 2017 data into one common column
    target_cols = []
    for c in h_cols:
        target_cols.extend([x for x in df_year.columns if c in x])
    df[h_cols] = df_year[target_cols]
    df['Happiness Score'] = df_year[[x for x in df_year.columns if 'Score' in x]]
    # Calculate quartiles on the data.
    df["target"] = pd.qcut(df[df.columns[-1]], len(targets), labels=targets)
    df["target_n"] = pd.qcut(df[df.columns[-2]], len(targets), labels=range(len(targets)))
    # Append year and assign to multi-index
    df['Year'] = year
    df = df.set_index(['Country', 'Year'])
    return df
df = prep_frame(df_2015, 2015)
df = df.append(prep_frame(df_2016, 2016), sort=False)
df = df.append(prep_frame(df_2017, 2017), sort=False)
# print('\n ********************Data in Tabular**************************** \n')
# print(df.head())
print('\n =============== Starting random forest regression =============================== \n')
#print('Size of df is :', df.info())
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
   df.drop(['target', 'target_n', 'Happiness Score'], axis='columns'), df['Happiness Score'], test_size=0.2)

# print('Size of X_train is :', X_train)
# print('Size of X_test is:', np.shape(X_test)[0])
model = sklearn.ensemble.RandomForestRegressor()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print('This is RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(Y_test, y_pred)))
print('The type of Y_train', np.shape(df['Happiness Score'].to_numpy()))
print('the type of y_pred',type(np.array(y_pred)))

##df.plot(x='Happiness Score', y=y_pred, kind = 'scatter')
plt.figure()
plt.scatter(np.array(y_pred), Y_test.to_numpy())
# plt.scatter(df['Happiness Score'].to_numpy().reshape, np.array(y_pred))
plt.show()


print('\n ================= Starting random forest classification =============== \n')
#print('Size of df is :', df.info())
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
   df.drop(['target', 'target_n', 'Happiness Score'], axis='columns'), df['target_n'], test_size=0.2)

# print('Size of X_train is :', X_train)
# print('Size of X_test is:', np.shape(X_test)[0])
model = sklearn.ensemble.RandomForestClassifier(n_estimators = 30)
model.fit(X_train, Y_train)
model.score(X_test,Y_test)
y_predicted = model.predict(X_test)
cm = sklearn.metrics.confusion_matrix(Y_test, y_predicted)
plt.figure()
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


