from django.apps import AppConfig


class SubscriptionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Subscriptions'
    def ready(self):
        import Subscriptions.signals
        