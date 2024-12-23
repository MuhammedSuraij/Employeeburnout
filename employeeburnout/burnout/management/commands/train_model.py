from django.core.management.base import BaseCommand
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Trains a linear regression model to predict employee burnout based on the provided dataset.'

    def add_arguments(self, parser):
        parser.add_argument('data_file', type=str, help='Path to the Excel dataset file.')

    def handle(self, *args, **kwargs):
        data_file = kwargs['data_file']

        if not os.path.exists(data_file):
            self.stdout.write(self.style.ERROR(f'The file {data_file} does not exist.'))
            return

        # Read the Excel data sheet
        data = pd.read_excel(data_file)

        # Drop rows with missing values
        data = data.dropna()

        # Drop 'Employee ID' column (as it has no correlation with Burn Rate)
        data = data.drop('Employee ID', axis=1)

        # Convert the 'Date of Joining' to days since '2008-01-01'
        yeardata = pd.to_datetime("2008-01-01")
        data["Days"] = (pd.to_datetime(data['Date of Joining']) - yeardata).dt.days

        # Calculate correlation between numeric columns and 'Burn Rate'
        numeric_data = data.select_dtypes(include=['number'])
        correlation = numeric_data.corr()['Burn Rate']
        self.stdout.write(self.style.SUCCESS(f'Correlation with Burn Rate:\n{correlation}'))

        # Drop 'Date of Joining' and 'Days' columns as their correlation with 'Burn Rate' is small
        data = data.drop(['Date of Joining', 'Days'], axis=1)

        # # Plotting count plots for string type columns
        # String_columns = data.select_dtypes(include=['object']).columns
        # fig, ax = plt.subplots(nrows=1, ncols=len(String_columns), sharey=True, figsize=(10, 5))
        # for i, c in enumerate(String_columns):
        #     sb.countplot(x=c, data=data, ax=ax[i])
        # plt.show()

        # Apply dummies for categorical variables
        if all(col in data.columns for col in ['Company Type', 'WFH Setup Available', 'Gender']):
            data = pd.get_dummies(data, columns=['Company Type', 'WFH Setup Available', 'Gender'], drop_first=True)

        # Preprocessing: Train-test split
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

        # Predictions
        y_pred = lr.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display the evaluation metrics
        # self.stdout.write(self.style.SUCCESS(f"Model Evaluation Results:"))
        # self.stdout.write(self.style.SUCCESS(f"Mean Absolute Error: {mae}"))
        # self.stdout.write(self.style.SUCCESS(f"Mean Squared Error: {mse}"))
        # self.stdout.write(self.style.SUCCESS(f"R² Score: {r2}"))

        # Optional: Save evaluation results to the database
        from burnout.models import Successrate
        model_eval = Successrate(mae=mae, mse=mse, r2=r2)
        model_eval.save()

        # self.stdout.write(self.style.SUCCESS(f'Evaluation results saved to the database.'))
