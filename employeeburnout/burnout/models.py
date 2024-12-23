from django.db import models

# Create your models here.
class Successrate(models.Model):
    mae= models.FloatField()
    mse=models.FloatField()
    r2=models.FloatField()
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Successrate{self.created_at}"
    
class BurnoutForm(models.Model):
    employee_id=models.CharField(max_length=100)
    date_of_joining=models.DateField()
    gender=models.CharField(max_length=10)
    company_type=models.CharField(max_length=10)
    wfh_setup=models.CharField(max_length=3)
    designation=models.IntegerField()
    resources_allocation=models.IntegerField()
    mental_fatigue_score=models.FloatField()

    def __str__(self):
        return f"Employee {self.employee_id}-Burnout data"