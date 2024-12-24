from django.urls import path
from .views import *

# urlpatterns = [
    
#     path('projects/', ProjectListView.as_view(), name='project-list'),
#     path('projects/create/', ProjectCreateView.as_view(), name='project-create'),
#     path('projects/<int:project_id>/working-drawing/', WorkingDrawingListView.as_view(), name='working-drawing-list'),
#     path('projects/<int:project_id>/working-drawing/create/', WorkingDrawingCreateView.as_view(), name='working-drawing-create'),
#     path('projects/<int:project_id>/plan/', PlanListView.as_view(), name='plan-list'),
#     path('projects/<int:project_id>/plan/create/', PlanCreateView.as_view(), name='plan-create'),
#     path('projects/<int:project_id>/3d/', ThreeDModelListView.as_view(), name='3d-model-list'),
#     path('projects/<int:project_id>/3d/create/', ThreeDModelCreateView.as_view(), name='3d-model-create'),
#     path('projects/<int:project_id>/concept/', ConceptListView.as_view(), name='concept-list'),
#     path('projects/<int:project_id>/concept/create/', ConceptCreateView.as_view(), name='concept-create'),
# ]
urlpatterns = [
    path('', ProcessDXFView.as_view(), name='workingdrawings'),
]