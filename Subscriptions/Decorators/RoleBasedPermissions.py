from functools import wraps
from django.core.exceptions import PermissionDenied
from django.utils import timezone
from Subscriptions.models import Subscription, CustomerProfile
import logging

logger = logging.getLogger(__name__)

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

                # Check storage limit
                storage_limit = subscription.plan.features.get('storage_limit_gb', 0)
                used_storage = request.user.customerprofile.used_storage_gb
                if used_storage >= storage_limit:
                    logger.warning(f"Access denied: storage limit exceeded for user {request.user.username}")
                    raise PermissionDenied("Storage limit exceeded")

                # Check project limit
                max_projects = subscription.plan.features.get('max_projects', 0)
                project_count = request.user.customerprofile.project_count
                if max_projects != -1 and project_count >= max_projects:
                    logger.warning(f"Access denied: project limit exceeded for user {request.user.username}")
                    raise PermissionDenied("Project limit exceeded")

                # Check file upload size
                if hasattr(request, 'FILES') and request.FILES:
                        total_upload_size = sum(f.size for f in request.FILES.values())
                        if used_storage + (total_upload_size / (1024 ** 3)) > storage_limit:
                            raise PermissionDenied("This upload would exceed your storage limit")
                            
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
