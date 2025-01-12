from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model
from django.forms import ValidationError
from Subscriptions.models import Subscription
from django.utils import timezone
from django.db.models import Sum    

class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='projects')

class SubProject(models.Model):
    project = models.ForeignKey(Project, related_name='subprojects', on_delete=models.CASCADE)
    type = models.CharField(max_length=20)
    state = models.JSONField(default=dict, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class BaseFileModel(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    size = models.PositiveIntegerField()  # Size in bytes
    created_at = models.DateTimeField(auto_now_add=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_size = self.size if self.id else None

    def save(self, *args, **kwargs):
        if self.id and self.size != self._original_size:
            # File is being updated, check storage limits
            subscription = Subscription.objects.filter(
                user=self.user,
                status='ACTIVE',
                end_date__gte=timezone.now()
            ).first()
            
            if subscription:
                storage_limit = subscription.plan.features.get('storage_limit_gb', 0) * (1024 ** 3)
                current_usage = BaseFileModel.objects.filter(
                    user=self.user
                ).exclude(id=self.id).aggregate(total=Sum('size'))['total'] or 0
                
                if current_usage + self.size > storage_limit:
                    raise ValidationError("Storage limit would be exceeded")
        
        super().save(*args, **kwargs)