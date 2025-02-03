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
    path('working-drawings/<int:pk>/', WorkingDrawingView.as_view(),name='working-drawing-detail'),
    path('projects/<int:project_id>/electrical/fixtures/', ElectricalFixturesView.as_view(), name='electrical-fixtures'),
    path('projects/<int:project_id>/electrical/fixtures/<int:pk>/', ElectricalFixturesView.as_view(), name='fixture-detail'),
    path('projects/<int:project_id>/wiring/', ElectricalWiringView.as_view(),name='electrical-wiring-list'),
    path('projects/<int:project_id>/wiring/<int:wiring_id>/',ElectricalWiringView.as_view(),name='electrical-wiring-detail'),
    path('projects/<int:project_id>/plumbing/watersupply/', WaterSupplyView.as_view(),name='water-supply-list'),
    path('projects/<int:project_id>/plumbing/watersupply/<int:supply_id>/',WaterSupplyView.as_view(),name='water-supply-detail'),
    path('projects/<int:project_id>/plumbing/complete/', PlumbingCompleteView.as_view(),name='plumbing-complete-list'),
    path('projects/<int:project_id>/plumbing/complete/<int:complete_id>/',PlumbingCompleteView.as_view(),name='plumbing-complete-detail'),
    path('projects/<int:project_id>/structure/main/',StructuralMainView.as_view(),name='plumbing-complete-list'),
    path('projects/<int:project_id>/structure/main/<int:main_id>',StructuralMainView.as_view(),name='plumbing-complete-list'),
]
