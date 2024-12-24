from django.apps import AppConfig


class DummyConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'FloorPlanning'
    def ready(self):
        import FloorPlanning.signals