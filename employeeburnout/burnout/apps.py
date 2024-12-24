from django.apps import AppConfig
from .train_burnout_model import BurnoutModel
from django.conf import settings
import os

class BurnoutConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'burnout'

    def ready(self):
        app_dir=os.path.dirname(__file__)
        data_file=os.path.join(app_dir,'employee_burnout_analysis-AI 2.xlsx')

        try:
            print("training th model")
            BurnoutModel.train_model(data_file)
            print("train complete")
        except Exception as e:
            print(f"Error occured:{str(e)}")
