# subscriptions/signals.py

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from project.models import  Project
from .models import CustomerProfile
from project.models import BaseFileModel
from FloorPlanning.models import UserFile
from django.contrib.auth import get_user_model

@receiver(post_save, sender=BaseFileModel)
@receiver(post_delete, sender=BaseFileModel)
def update_storage_usage(sender, instance, **kwargs):
    user_profile = CustomerProfile.objects.get(user=instance.user)
    total_usage = sum(file.size for file in BaseFileModel.objects.filter(user=instance.user))
    user_profile.used_storage_gb = total_usage / (1024 ** 3)  # Convert bytes to GB
    user_profile.save()

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