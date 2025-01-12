from datetime import timezone
from rest_framework import serializers
from .models import User, Contact
from django.utils.encoding import smart_str, force_bytes, DjangoUnicodeDecodeError
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from account.utils import Util
from django.core.cache import cache
from Subscriptions.models import CustomerProfile, Subscription
import random 
from django.contrib.auth import authenticate, logout

class UserRegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    name = serializers.CharField(required=True)
    password = serializers.CharField(write_only=True, required=True)
    otp = serializers.CharField(required=False)

    def validate(self, data):
        return data

    def create(self, validated_data):
        otp = validated_data.pop('otp', None)
        email = validated_data.get('email')

        # Check if OTP is valid
        stored_otp = cache.get(f'registration_otp_{email}')
        if otp and otp != stored_otp:
            raise serializers.ValidationError("Invalid OTP")

        # Clear the OTP from cache
        if otp:
            cache.delete(f'registration_otp_{email}')

        return User.objects.create_user(**validated_data)

    @staticmethod
    def generate_otp(email):
        otp = str(random.randint(100000, 999999))
        cache.set(f'registration_otp_{email}', otp, timeout=300)  # OTP valid for 5 minutes
        return otp


class UserLoginSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(max_length=255)
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['email', 'password']

    def validate(self, attrs):
        """
        Override the default validate method to handle user authentication.
        """
        email = attrs.get('email')
        password = attrs.get('password')

        # Authenticate the user
        user = authenticate(email=email, password=password)

        if not user:
            raise serializers.ValidationError("Invalid email or password.")

        # Add the authenticated user to validated_data so it can be used in the view
        attrs['user'] = user
        return attrs

    def to_representation(self, instance):
        """
        Override the to_representation method to include customer type and subscription info.
        """
        data = super().to_representation(instance)

        try:
            customer_profile = CustomerProfile.objects.get(user=instance)
            active_subscription = Subscription.objects.filter(
                user=instance,
                status='ACTIVE',
                end_date__gte=timezone.now()
            ).first()

            data['customer_type'] = customer_profile.customer_type

            if active_subscription:
                data['subscription'] = {
                    'plan': active_subscription.plan.name,
                    'billing_cycle': active_subscription.plan.billing_cycle,
                    'end_date': active_subscription.end_date
                }

        except CustomerProfile.DoesNotExist:
            pass  # If no profile is found, we do nothing, and customer_type will be None

        return data

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'name', 'phone', 'location']

class UserChangePasswordSerializer(serializers.Serializer):
    password = serializers.CharField(max_length=255, style={'input_type': 'password'}, write_only=True)
    password2 = serializers.CharField(max_length=255, style={'input_type': 'password'}, write_only=True)
    
    class Meta:
        fields = ['password', 'password2']

    def validate(self, attrs):
        password = attrs.get('password')
        password2 = attrs.get('password2')
        user = self.context.get('user')
        if password != password2:
            raise serializers.ValidationError("Password and Confirm Password doesn't match")
        user.set_password(password)
        user.save()
        return attrs

class SendPasswordResetEmailSerializer(serializers.Serializer):
    email = serializers.EmailField(max_length=255)
    
    class Meta:
        fields = ['email']

    def validate(self, attrs):
        email = attrs.get('email')
        if User.objects.filter(email=email).exists():
            user = User.objects.get(email=email)
            uid = urlsafe_base64_encode(force_bytes(user.id))
            print('Encoded UID', uid)
            token = PasswordResetTokenGenerator().make_token(user)
            print('Password Reset Token', token)
            link = f'http://localhost:3000/api/user/reset/{uid}/{token}'
            print('Password Reset Link', link)
            # Send Email
            body = f'Click Following Link to Reset Your Password {link}'
            data = {
                'subject': 'Reset Your Password',
                'body': body,
                'to_email': user.email
            }
            # Util.send_email(data)
            return attrs
        else:
            raise serializers.ValidationError('You are not a Registered User')

class UserPasswordResetSerializer(serializers.Serializer):
    password = serializers.CharField(max_length=255, style={'input_type': 'password'}, write_only=True)
    password2 = serializers.CharField(max_length=255, style={'input_type': 'password'}, write_only=True)
    
    class Meta:
        fields = ['password', 'password2']

    def validate(self, attrs):
        try:
            password = attrs.get('password')
            password2 = attrs.get('password2')
            uid = self.context.get('uid')
            token = self.context.get('token')
            if password != password2:
                raise serializers.ValidationError("Password and Confirm Password doesn't match")
            id = smart_str(urlsafe_base64_decode(uid))
            user = User.objects.get(id=id)
            if not PasswordResetTokenGenerator().check_token(user, token):
                raise serializers.ValidationError('Token is not Valid or Expired')
            user.set_password(password)
            user.save()
            return attrs
        except DjangoUnicodeDecodeError as identifier:
            PasswordResetTokenGenerator().check_token(user, token)
            raise serializers.ValidationError('Token is not Valid or Expired')

class ContactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contact
        fields = ['id', 'name', 'contact', 'email', 'message']

class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()

class ResetPasswordSerializer(serializers.Serializer):
    password = serializers.CharField(min_length=6, max_length=68, write_only=True)
    password2 = serializers.CharField(min_length=6, max_length=68, write_only=True)

    def validate(self, attrs):
        password = attrs.get('password')
        password2 = attrs.get('password2')
        if password != password2:
            raise serializers.ValidationError("Passwords do not match")
        return attrs
