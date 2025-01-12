from django.urls import path
from .views import *

urlpatterns = [
    # SubscriptionPlan URLs
    path('plans/', SubscriptionPlanListView.as_view(), name='subscription-plan-list'),
    path('plans/<int:pk>/', SubscriptionPlanDetailView.as_view(), name='subscription-plan-detail'),
    
    # CustomerProfile URLs
    path('customer-profile/', CustomerProfileView.as_view(), name='customer-profile'),
    
    # Subscription URLs
    path('subscriptions/', SubscriptionListView.as_view(), name='subscription-list'),
    path('subscriptions/<int:pk>/', SubscriptionDetailView.as_view(), name='subscription-detail'),

    # Storage Checker
    path('storagecheck/', StorageUsageView.as_view(), name='storage-check'),
]
