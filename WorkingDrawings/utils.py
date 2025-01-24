
from project.models import SubProject
from django.utils import timezone

def get_or_create_working_drawings_subproject(project):
    """
    Gets existing working drawings subproject or creates a new one.
    All working drawings features (electrical, plumbing, etc.) will be linked to this.
    """
    working_drawings = SubProject.objects.filter(
        project=project,
        type='working_drawings'
    ).first()
    
    if not working_drawings:
        working_drawings = SubProject.objects.create(
            project=project,
            type='working_drawings',
            state={
                'status': 'active',
                'features': {
                    'electrical': {'status': 'pending', 'count': 0},
                    'plumbing': {'status': 'pending', 'count': 0},
                    'structural': {'status': 'pending', 'count': 0}
                },
                'last_modified': timezone.now().isoformat()
            }
        )
    return working_drawings