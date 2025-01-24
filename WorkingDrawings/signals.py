from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import *
from django.utils import timezone

@receiver(post_save, sender=SubProject)
def create_working_drawing(sender, instance, created, **kwargs):
    if created and instance.type == 'working_drawing':
        WorkingDrawingProject.objects.create(subproject=instance)

@receiver(post_save, sender=ElectricalFixtures)
@receiver(post_save, sender=ElectricalWiring)
def update_electrical_status(sender, instance, created, **kwargs):
    """Update electrical section status when fixtures/wiring status changes"""
    if not created and instance.status in ['completed', 'failed']:
        working_drawing = instance.subproject.working_drawing
        if working_drawing:
            # Get all electrical components
            fixtures = ElectricalFixtures.objects.filter(
                subproject=instance.subproject
            )
            wiring = ElectricalWiring.objects.filter(
                subproject=instance.subproject
            )
            
            # Calculate overall status
            all_completed = all(
                item.status == 'completed' 
                for item in list(fixtures) + list(wiring)
                if item.status != 'failed'
            )
            
            has_failed = any(
                item.status == 'failed'
                for item in list(fixtures) + list(wiring)
            )
            
            # Determine section status
            if has_failed:
                new_status = 'failed'
            elif all_completed:
                new_status = 'completed'
            else:
                new_status = 'in_progress'
            
            # Update working drawing state
            state = instance.subproject.state
            state['sections']['electrical'].update({
                'status': new_status,
                'last_updated': timezone.now().isoformat()
            })
            
            # Update overall project status
            all_sections_completed = all(
                section['status'] == 'completed'
                for section in state['sections'].values()
            )
            state['status'] = 'completed' if all_sections_completed else 'in_progress'
            
            instance.subproject.state = state
            instance.subproject.save()      