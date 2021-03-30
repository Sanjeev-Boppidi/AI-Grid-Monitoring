from django.urls import path
from . import views

urlpatterns = [
    path('index', views.index,name='index'),
    path('dashboard', views.dashboard,name='dashboard'),
    path('fault', views.fault,name='fault'),
    path('forecasting', views.forecasting,name='forecasting'),
    path('dataanalysis1', views.dataanalysis1,name='dataanalysis1'),
    path('dataanalysis2', views.dataanalysis2,name='dataanalysis2'),
]