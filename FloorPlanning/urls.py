from django.urls import path
from .views import *


urlpatterns = [
    path('create-project/', CreateProjectView.as_view(), name='create-project'),
    path('pdf-list/', UserFileListView.as_view(), name='pdf-list'),
    path('generate-map-soil-data/', GenerateMapAndSoilDataView.as_view(), name='generate-map-soil-data'),
]
