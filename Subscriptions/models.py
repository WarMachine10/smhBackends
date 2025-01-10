from django.utils import timezone
from django.db import models
from django.db import models
from account.models import User
from django.core.exceptions import ValidationError
# Create your models here.
class SubscriptionPlan(models.Model):
    PLAN_TYPES = (
        ('PREMIUM', 'Premium'),
        ('GROWTH', 'Growth'),
        ('ELITE', 'Elite')
    )
    BILLING_CYCLES = (
        ('MONTHLY', 'Monthly'),
        ('YEARLY', 'Yearly')
    )
    name = models.CharField(max_length=50, choices=PLAN_TYPES)
    billing_cycle = models.CharField(max_length=10, choices=BILLING_CYCLES)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    features = models.JSONField()  # Store features as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('name', 'billing_cycle')

class CustomerProfile(models.Model):
    CUSTOMER_TYPES = (
        ('B2B', 'Business'),
        ('B2C', 'Consumer')
    )
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    customer_type = models.CharField(max_length=3, choices=CUSTOMER_TYPES)
    company_name = models.CharField(max_length=255, null=True, blank=True)
    tax_id = models.CharField(max_length=50, null=True, blank=True)
    
    def clean(self):
        if self.customer_type == 'B2B' and not self.company_name:
            raise ValidationError("Company name is required for B2B customers")

class Subscription(models.Model):
    STATUS_CHOICES = (
        ('ACTIVE', 'Active'),
        ('CANCELLED', 'Cancelled'),
        ('EXPIRED', 'Expired'),
        ('PENDING', 'Pending')
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    payment_id = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.plan.name} - {self.status}"

    def clean(self):
        # Check for overlapping active subscriptions
        if self.status == 'ACTIVE':
            overlapping = Subscription.objects.filter(
                user=self.user,
                status='ACTIVE',
                end_date__gt=timezone.now()
            ).exclude(pk=self.pk)
            if overlapping.exists():
                raise ValidationError("User already has an active subscription")