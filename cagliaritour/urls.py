from django.urls import path
from .views import *
urlpatterns = [
    path('getroute/<str:numberofdays>/', calculate_route, name='getroute'),
   ]