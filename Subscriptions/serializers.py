# subscriptions/serializers.py
from rest_framework import serializers
from .models import SubscriptionPlan, CustomerProfile, Subscription

class SubscriptionPlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubscriptionPlan
        fields = '__all__'

class CustomerProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerProfile
        fields = ['customer_type', 'company_name', 'tax_id']

    def validate(self, data):
        if data.get('customer_type') == 'B2B' and not data.get('company_name'):
            raise serializers.ValidationError(
                {"company_name": "Company name is required for B2B customers"}
            )
        return data


class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = '__all__'
        read_only_fields = ('user', 'status', 'payment_id','start_date', 'end_date' )


        