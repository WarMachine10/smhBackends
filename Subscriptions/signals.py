# subscriptions/signals.py

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from project.models import  Project
from .models import CustomerProfile
from project.models import BaseFileModel
from FloorPlanning.models import UserFile
from django.contrib.auth import get_user_model
from loguru import logger
from Subscriptions.models import Subscription
from django.utils import timezone
from django.db.models import Sum
from django.core.exceptions import ValidationError

@receiver(post_save, sender=BaseFileModel)
def update_storage_on_save(sender, instance, created, **kwargs):
    try:
        user_profile = CustomerProfile.objects.select_for_update().get(user=instance.user)
        
        # Get active subscription
        subscription = Subscription.objects.filter(
            user=instance.user,
            status='ACTIVE',
            end_date__gte=timezone.now()
        ).first()
        
        if not subscription:
            raise ValidationError("No active subscription found")
            
        storage_limit = subscription.plan.features.get('storage_limit_gb', 0) * (1024 ** 3)  # Convert GB to bytes
        
        # Calculate total storage including the new file
        total_usage = BaseFileModel.objects.filter(user=instance.user).aggregate(
            total=Sum('size'))['total'] or 0
            
        # For updates, subtract the old file size if it exists
        if not created and hasattr(instance, '_original_size'):
            total_usage -= instance._original_size
            
        # Check if adding this file would exceed the limit
        if total_usage > storage_limit:
            raise ValidationError(
                f"Storage limit exceeded. Limit: {storage_limit/(1024**3):.2f}GB, "
                f"Used: {total_usage/(1024**3):.2f}GB"
            )
            
        # Update storage usage in GB
        user_profile.used_storage_gb = total_usage / (1024 ** 3)
        user_profile.save(update_fields=['used_storage_gb'])
        
        # Store original size for future updates
        instance._original_size = instance.size
        
    except CustomerProfile.DoesNotExist:
        logger.error(f"Customer profile not found for user {instance.user.id}")
        raise

@receiver(post_delete, sender=BaseFileModel)
def update_storage_on_delete(sender, instance, **kwargs):
    try:
        user_profile = CustomerProfile.objects.select_for_update().get(user=instance.user)
        
        # Recalculate total storage excluding the deleted file
        total_usage = BaseFileModel.objects.filter(user=instance.user).exclude(
            id=instance.id).aggregate(total=Sum('size'))['total'] or 0
            
        user_profile.used_storage_gb = total_usage / (1024 ** 3)
        user_profile.save(update_fields=['used_storage_gb'])
        
    except CustomerProfile.DoesNotExist:
        logger.error(f"Customer profile not found for user {instance.user.id}")
        raise

@receiver(post_save, sender=Project)
@receiver(post_delete, sender=Project)
def update_project_count(sender, instance, **kwargs):
    user_profile = CustomerProfile.objects.get(user=instance.user)
    user_profile.project_count = Project.objects.filter(user=instance.user).count()
    user_profile.save()

@receiver(post_save, sender=get_user_model())
def create_customer_profile(sender, instance, created, **kwargs):
    if created:
        CustomerProfile.objects.create(user=instance)