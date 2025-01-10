from Subscriptions.models import Subscription
from datetime import timezone
class SubscriptionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            subscription = getattr(request, 'subscription', None)
            if not subscription:
                subscription = Subscription.objects.filter(
                    user=request.user,
                    status='ACTIVE',
                    end_date__gte=timezone.now()
                ).first()
                request.subscription = subscription
        return self.get_response(request)