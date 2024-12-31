from django.urls import path
from .views import *

urlpatterns = [
    path('', ProjectListCreateView.as_view(), name='project-list-create'),
    path('<int:pk>/', ProjectRetrieveUpdateDestroyView.as_view(), name='project-detail'),
    path('subprojects/', SubProjectListCreateView.as_view(), name='subproject-list-create'),
    path('subprojects/<int:pk>/', SubProjectRetrieveUpdateDestroyView.as_view(), name='subproject-detail'),
]