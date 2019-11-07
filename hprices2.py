import pandas as pd
import numpy as np

file_dir = "input/house-prices/"

train = pd.read_csv(file_dir+"train.csv")
test = pd.read_csv(file_dir+"test.csv")
train.name = "TRAINING"
test.name = "TESTING"
##########################################################################CLEANING#############################

def onehot2(dataframe = pd.DataFrame(), series = pd.Series()):
    #ONEHOTS SERIES WITH NON NUMERICAL OR NUMERICAL INFORMATION THAT REFERS TO QUALITATIVE VALUES
    onehot_series = []
    base_name = series.name #to identify which series the onehot comes from
    unique_values = series.unique()
    iteration = 0
    for unique_value in unique_values:  #creates new series for every unique value
        onehotSeries = pd.Series()
        onehotSeries.name = base_name + str(iteration)  
        for value in series.iteritems():    #fills the new dummy series with binary float values to reflect which rows have that feature
            if unique_value == value[1]:
                onehotSeries.at[value[0]] = 1.0
            else:
                onehotSeries.at[value[0]] = 0.0
        iteration = iteration + 1
        onehot_series.append(onehotSeries)  #adds to the list of onehots to be added to the dataframe
        del onehotSeries        
    for i in onehot_series:     #adds new binary onehot dummy series to the main dataframe
        dataframe[i.name] = i
    if not {series.name}.issubset(dataframe.columns):   #if the series was not originially in the input dataframe ignore and return the new/used dataframe
        return dataframe
    else:                                               #returns the dataframe with the onehotted series dropped from records for processing
        dataframe.drop(series.name, axis=1, inplace=True)
        return dataframe

def FindIrrelevent(dataframe = pd.DataFrame(), threshold = 1):
    #IDENTIFIES DUMB and irrelevant INFORMATION
    irrelevent = []
    for series in dataframe.columns:
        if len(dataframe[series].unique()) <= threshold:
            irrelevent.append(series)
        else:
            continue
    return irrelevent

def FindOneHot(dataframe = pd.DataFrame(), threshold = 40):
    #IDENTIFIES POTENTIAL ONEHOT SERIES IN THE DATAFRAME SHEET
    #threshold is an arbitary number below the number of max rows
    recommendation = []
    for series in dataframe.columns:
        if len(dataframe[series].unique()) <= threshold:
            if len(dataframe[series].unique()) > 1:
                recommendation.append(series)
        else:
            continue
    return recommendation

def imputer(dataframe = pd.DataFrame(), columns = []):
    for series in columns:
       for value in dataframe[series].iteritems():
           if type(value[1]) != int or type(value[1]) != float: 
               dataframe[series].at[value[0]] = dataframe[series].median()
    return dataframe

def clean(label = None, dataframe = pd.DataFrame(), droplist= [], onehot_list = [], imputer_list = []):

    #generating onehots
    for x in onehot_list:
        dataframe = onehot2(dataframe, dataframe[x])
        print("FINISHED ONEHOTTING:\t"+ x + "\tin\t" + dataframe.name)
    #dropping
    for pledge in droplist:
        dataframe.drop(pledge, axis=1, inplace=True)
    #features v labels
    y = 0
    if label is not None:
        y = dataframe[label]
        dataframe.drop(label, axis=1, inplace=True)
    dataframe.dropna()
    return dataframe, y

########################################################################REGRESSION##################################
imputer_list = [""]
droplist = ["Id"]

train_x, train_y = clean("SalePrice", dataframe = train, droplist = droplist, onehot_list=FindOneHot(dataframe=train))
test_x, _ = clean(dataframe = test, droplist = droplist, onehot_list=FindOneHot(dataframe=test))

train_x.to_csv("train_x-house.csv")
train_y.to_csv("train_y-house.csv")
test_x.to_csv("test_x-house.csv")

#import tensorflow as tf
#import tensorboard as tb

#print(tf.__version__)
"""
def sklearnRegressionModel(Training, Validation):
    from sklearn.linear_model import LinearRegression
    import sklearn.metrics as metrics
    model = LinearRegression()
    model.fit(Training[0], Training[1])
    model.score(Training[0], Training[1])
    y_prediction =model.predict(Validation[0])
    rmse=np.sqrt(metrics.mean_squared_error(y_prediction, Validation[1]))

    return model, rmse

_, rmse = sklearnRegressionModel([train_x, train_y], [test_x, test_y])
print(rmse)"""