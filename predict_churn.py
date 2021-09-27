import pandas as pd
from pycaret.classification import predict_model, load_model

#Set Variable
Path_=input("Enter dataset: ")

def load_data(filepath):
#Create Dataframe from .csv input
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
#Use what was determined to be the best model to make predictions on the df 
    model = load_model(r'C:\Users\Schry\Documents\MSDS600\Week 5/GBC')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
    return predictions['Churn_prediction']


if __name__ == "__main__":
    df = load_data(Path_)
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)