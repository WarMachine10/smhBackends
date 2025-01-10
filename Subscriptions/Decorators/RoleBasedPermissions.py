from datetime import timezone
from functools import wraps
from rest_framework.exceptions import PermissionDenied
from ..models import Subscription, CustomerProfile
from loguru import logger
from functools import wraps
from rest_framework.exceptions import PermissionDenied
from django.utils import timezone
def subscription_required(plan_names=None, customer_types=None):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            # Check customer type if specified
            if customer_types:
                try:
                    profile = request.user.customerprofile
                    if profile.customer_type not in customer_types:
                        logger.warning(f"Access denied: wrong customer type for user {request.user.username}")
                        raise PermissionDenied("Feature not available for your customer type")
                except CustomerProfile.DoesNotExist:
                    logger.warning(f"Access denied: no customer profile for user {request.user.username}")
                    raise PermissionDenied("Customer profile not found")

            # Check subscription plan if specified
            if plan_names:
                subscription = Subscription.objects.filter(
                    user=request.user,
                    status='ACTIVE',
                    end_date__gte=timezone.now(),
                    plan__name__in=plan_names
                ).first()
                
                if not subscription:
                    logger.warning(f"Access denied: no active subscription for user {request.user.username}")
                    raise PermissionDenied("Feature requires an active subscription")

            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

def customer_type_required(customer_types):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            try:
                profile = CustomerProfile.objects.get(user=request.user)
                if profile.customer_type not in customer_types:
                    logger.warning(f"Access denied: wrong customer type for user {request.user.username}")
                    raise PermissionDenied("This feature is not available for your customer type")
                return view_func(request, *args, **kwargs)
            except CustomerProfile.DoesNotExist:
                logger.warning(f"Access denied: no customer profile for user {request.user.username}")
                raise PermissionDenied("Customer profile not found")
        return wrapper
    return decorator