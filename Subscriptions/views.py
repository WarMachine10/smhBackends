# subscriptions/views.py
from rest_framework import viewsets, status, mixins
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound
from datetime import datetime, timedelta
from .models import SubscriptionPlan, CustomerProfile, Subscription
from .serializers import SubscriptionPlanSerializer, CustomerProfileSerializer, SubscriptionSerializer
from rest_framework_simplejwt.authentication import JWTAuthentication
from loguru import logger
from django.utils import timezone
from django.core.exceptions import PermissionDenied

class SubscriptionPlanListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, *args, **kwargs):
        plans = SubscriptionPlan.objects.all()
        serializer = SubscriptionPlanSerializer(plans, many=True)
        return Response(serializer.data)
    
    

class SubscriptionPlanDetailView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, pk, *args, **kwargs):
        try:
            plan = SubscriptionPlan.objects.get(pk=pk)
            serializer = SubscriptionPlanSerializer(plan)
            return Response(serializer.data)
        except SubscriptionPlan.DoesNotExist:
            raise NotFound("Subscription Plan not found")

    def post(self, request, *args, **kwargs):
        # Add admin check
        if not request.user.is_staff:
            raise PermissionDenied("Only administrators can create subscription plans")
        
        serializer = SubscriptionPlanSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            logger.info(f"Created new subscription plan: {serializer.data['name']}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk, *args, **kwargs):
        if not request.user.is_staff:
            raise PermissionDenied("Only administrators can update subscription plans")
            
        try:
            plan = SubscriptionPlan.objects.get(pk=pk)
            serializer = SubscriptionPlanSerializer(plan, data=request.data)
            if serializer.is_valid():
                serializer.save()
                logger.info(f"Updated subscription plan: {plan.name}")
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except SubscriptionPlan.DoesNotExist:
            raise NotFound("Subscription Plan not found")

    def delete(self, request, pk, *args, **kwargs):
        if not request.user.is_staff:
            raise PermissionDenied("Only administrators can delete subscription plans")
            
        try:
            plan = SubscriptionPlan.objects.get(pk=pk)
            # Check if plan has active subscriptions
            if Subscription.objects.filter(plan=plan, status='ACTIVE').exists():
                return Response(
                    {"error": "Cannot delete plan with active subscriptions"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            plan.delete()
            logger.info(f"Deleted subscription plan: {plan.name}")
            return Response(status=status.HTTP_204_NO_CONTENT)
        except SubscriptionPlan.DoesNotExist:
            raise NotFound("Subscription Plan not found")

class CustomerProfileView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, *args, **kwargs):
        try:
            profile = CustomerProfile.objects.get(user=request.user)
            serializer = CustomerProfileSerializer(profile)
            return Response(serializer.data)
        except CustomerProfile.DoesNotExist:
            raise NotFound("Customer Profile not found")

    def post(self, request, *args, **kwargs):
        try:
            # Check if profile already exists
            if hasattr(request.user, 'customerprofile'):
                return Response(
                    {"error": "Customer profile already exists"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            serializer = CustomerProfileSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save(user=request.user)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def put(self, request, *args, **kwargs):
        try:
            profile = CustomerProfile.objects.get(user=request.user)
            serializer = CustomerProfileSerializer(profile, data=request.data)
            if serializer.is_valid():
                serializer.save()
                logger.info(f"Updated customer profile for user {request.user.username}")
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except CustomerProfile.DoesNotExist:
            raise NotFound("Customer Profile not found")

class SubscriptionListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, *args, **kwargs):
        subscriptions = Subscription.objects.filter(user=request.user)
        serializer = SubscriptionSerializer(subscriptions, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        """Create a new subscription"""
        # Check for existing active subscription
        active_subscription = Subscription.objects.filter(
            user=request.user,
            status='ACTIVE',
            end_date__gt=timezone.now()
        ).first()

        if active_subscription:
            return Response(
                {"error": "Active subscription already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = SubscriptionSerializer(data=request.data)
        if serializer.is_valid():
            plan = serializer.validated_data['plan']
            
            # Calculate dates
            start_date = timezone.now()
            if plan.billing_cycle == 'MONTHLY':
                end_date = start_date + timedelta(days=30)
            else:  # YEARLY
                end_date = start_date + timedelta(days=365)
            
            # Create subscription with calculated dates
            subscription = serializer.save(
                user=request.user,
                start_date=start_date,
                end_date=end_date,
                status='ACTIVE'
            )
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SubscriptionDetailView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, pk, *args, **kwargs):
        try:
            subscription = Subscription.objects.get(pk=pk, user=request.user)
            serializer = SubscriptionSerializer(subscription)
            return Response(serializer.data)
        except Subscription.DoesNotExist:
            raise NotFound("Subscription not found")

    def post(self, request, *args, **kwargs):
        """Create a new subscription"""
        # Check for existing active subscription
        active_subscription = Subscription.objects.filter(
            user=request.user,
            status='ACTIVE',
            end_date__gt=timezone.now()
        ).first()

        if active_subscription:
            return Response(
                {"error": "Active subscription already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = SubscriptionSerializer(data=request.data)
        if serializer.is_valid():
            plan = serializer.validated_data['plan']
            
            # Dummy payment processing (to be replaced with actual gateway)
            dummy_payment_id = f"dummy_payment_{timezone.now().timestamp()}"
            
            # Set subscription dates
            start_date = timezone.now()
            if plan.billing_cycle == 'MONTHLY':
                end_date = start_date + timedelta(days=30)
            else:  # YEARLY
                end_date = start_date + timedelta(days=365)
            
            # Create subscription
            subscription = serializer.save(
                user=request.user,
                start_date=start_date,
                end_date=end_date,
                status='ACTIVE',
                payment_id=dummy_payment_id
            )
            
            logger.info(f"Created subscription for user {request.user.username}: {subscription}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk, *args, **kwargs):
        """Update subscription (e.g., for upgrades/downgrades)"""
        try:
            subscription = Subscription.objects.get(pk=pk, user=request.user)
            
            # Only allow updates to active subscriptions
            if subscription.status != 'ACTIVE':
                return Response(
                    {"error": "Can only update active subscriptions"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            serializer = SubscriptionSerializer(subscription, data=request.data, partial=True)
            if serializer.is_valid():
                updated_subscription = serializer.save()
                logger.info(f"Updated subscription for user {request.user.username}: {updated_subscription}")
                return Response(serializer.data)
            
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Subscription.DoesNotExist:
            raise NotFound("Subscription not found")

    def delete(self, request, pk, *args, **kwargs):
        """Cancel subscription"""
        try:
            subscription = Subscription.objects.get(pk=pk, user=request.user)
            
            if subscription.status != 'ACTIVE':
                return Response(
                    {"error": "Can only cancel active subscriptions"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            subscription.status = 'CANCELLED'
            subscription.save()
            
            logger.info(f"Cancelled subscription for user {request.user.username}: {subscription}")
            return Response({'status': 'Subscription cancelled'}, status=status.HTTP_200_OK)
        except Subscription.DoesNotExist:
            raise NotFound("Subscription not found")