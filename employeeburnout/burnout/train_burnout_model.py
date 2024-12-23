# train_burnout_model.py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from django.conf import settings

class BurnoutModel:
    _model = None
    _scaler = None
    _train_column= None

    @classmethod 
    def train_model(cls, data_file):
        # Load the data
        data = pd.read_excel('burnout\employee_burnout_analysis-AI 2.xlsx')
        data = data.dropna()  # Drop missing values
        data = data.drop('Employee ID', axis=1)  # Drop 'Employee ID' column

        # Convert 'Date of Joining' to days since 2008-01-01
        yeardata = pd.to_datetime("2008-01-01")
        data["Days"] = (pd.to_datetime(data['Date of Joining']) - yeardata).dt.days
        data = data.drop(['Date of Joining', 'Days'], axis=1)

        # Apply dummies for categorical variables
        data=pd.get_dummies(data,columns=['Company Type','WFH Setup Available','Gender'],drop_first=True)
        
        cls._train_column=data.columns

        # Train-test split
        y = data['Burn Rate']
        X = data.drop('Burn Rate', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=55)

        # Scaling the data
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        # Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Save the model and scaler to the class
        cls._model = lr
        cls._scaler = scaler

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise Exception("Model is not trained yet!")
        return cls._model

    @classmethod
    def get_scaler(cls):
        if cls._scaler is None:
            raise Exception("Scaler is not trained yet!")
        return cls._scaler
