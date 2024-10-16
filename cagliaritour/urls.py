from django.urls import path
from .views import *
urlpatterns = [
    path('getroute/<str:numberofdays>/', calculate_route, name='getroute'),
    path('addToMainTravelList/', add_to_main_travel_list, name='add_to_main_travel_list'),
    path('removeFromMainTravelList/', remove_from_main_travel_list, name='remove_from_main_travel_list'),

]