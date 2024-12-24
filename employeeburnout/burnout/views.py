from django.shortcuts import render,redirect
import pandas as pd
from .train_burnout_model import BurnoutModel
from django.http import HttpResponseBadRequest
from .models import BurnoutForm


# Create your views here.
def indexview(request):
    return render (request,'index.html')

# def formview(request):
#     return render(request,'form.html')
def about(request):
    return render(request,'about.html')


def burnout_calculator(request):
    if request.method == 'POST':
        # Collect form data
        form_data = {
            'Gender': request.POST.get('gender'),
            'Company Type': request.POST.get('company_type'),
            'WFH Setup Available': request.POST.get('wfh_setup'),
            'Designation': request.POST.get('designation'),
            'Resource Allocation': request.POST.get('resources_allocation'),
            'Mental Fatigue Score': request.POST.get('mental_fatigue_score')
        }

        # Create DataFrame from the form data
        input_df = pd.DataFrame([form_data])

        # One-hot encode the 'Company Type' and other categorical columns
        input_df = pd.get_dummies(input_df, columns=['Company Type', 'WFH Setup Available', 'Gender'], prefix=['Company Type', 'WFH Setup Available', 'Gender'])

        # Ensure that both columns for 'Company Type' exist (even if one has value 0)
        if 'Company Type_Service' not in input_df.columns:
            input_df['Company Type_Service'] = 0
        if 'Company Type_Product' not in input_df.columns:
            input_df['Company Type_Product'] = 0
        
        if 'WFH Setup Available_Yes' not in input_df.columns:
            input_df['WFH Setup Available_Yes'] = 0
        if 'WFH Setup Available_No' not in input_df.columns:
            input_df['WFH Setup Available_No'] = 0
       
        if 'Gender_Male' not in input_df.columns:
            input_df['Gender_Male'] = 0
        if 'Gender_Female' not in input_df.columns:
            input_df['Gender_Female'] = 0


        input_df = input_df.drop('Gender_Female', axis=1)
        input_df = input_df.drop('WFH Setup Available_No', axis=1)
        input_df = input_df.drop('Company Type_Product', axis=1)



        # Print to check the dataframe structure after encoding
        print(input_df)

        # Get the model and scaler from the BurnoutModel
        model = BurnoutModel.get_model()
        scalar = BurnoutModel.get_scaler()

        # Apply scaling to the input features
        input_df_scaled = scalar.transform(input_df)

        # Get the model's prediction
        prediction = model.predict(input_df_scaled)

        percentage = prediction[0] * 100
        # Render the prediction in the response
        return render(request, 'prediction.html', {'percentage': percentage,'prediction': prediction[0]})

    # If it's a GET request, render the form
    return render(request, 'form.html')

    