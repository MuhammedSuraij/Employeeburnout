from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('',views.indexview,name='home'),
    path('burnout_test',views.burnout_calculator,name='burnout_test'),
    path('about',views.about,name='about')


]