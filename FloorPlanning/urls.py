from django.urls import path
from .views import *

urlpatterns = [
    # Keep your existing views but organize paths better
    path('<int:project_id>/projects/', CreateProjectView.as_view(), name='create-project'),
    path('projects/<int:project_id>/files/', UserFileListView.as_view(), name='pdf-list'),
    path('projects/<int:project_id>/map-soil-data/', GenerateMapAndSoilDataView.as_view(), name='generate-map-soil-data'),
]
