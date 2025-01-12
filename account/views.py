from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from account.serializers import *
from django.contrib.auth import authenticate, logout
from account.renderers import UserRenderer
import boto3,os,datetime
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_str
from sib_api_v3_sdk import Configuration, ApiClient, TransactionalEmailsApi, SendSmtpEmail
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken
from django.urls import reverse
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from rest_framework.permissions import IsAuthenticated
from .models import User
from django.conf import settings
from .utils import Util
from loguru import logger
from drf_yasg.utils import swagger_auto_schema

load_dotenv()

# Generate Token Manually

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)

    # Get customer profile info
    try:
        customer_profile = CustomerProfile.objects.get(user=user)
        customer_type = customer_profile.customer_type
    except CustomerProfile.DoesNotExist:
        customer_type = None

    # Get subscription info
    try:
        active_subscription = Subscription.objects.filter(
            user=user,
            status='ACTIVE',
            end_date__gte=timezone.now()
        ).select_related('plan').first()

        subscription_info = {
            'plan': active_subscription.plan.name,
            'billing_cycle': active_subscription.plan.billing_cycle,
            'end_date': str(active_subscription.end_date)
        } if active_subscription else None
    except:
        subscription_info = None

    # Add custom claims to token
    refresh.payload.update({
        'email': user.email,
        'name': user.name,
        'customer_type': customer_type,
        'subscription': subscription_info
    })

    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

def send_brevo_email(to_email, to_name, subject, html_content):
    configuration = Configuration()
    configuration.api_key['api-key'] = settings.EMAIL_XAPI
    api_client = ApiClient(configuration)
    api_instance = TransactionalEmailsApi(api_client)

    send_smtp_email = SendSmtpEmail(
        to=[{"email": to_email, "name": to_name}],
        subject=subject,
        html_content=html_content,
        sender={"email": "server@sketchmyhome.ai", "name": "SketchMyHome.ai"}
    )
    try:
        api_instance.send_transac_email(send_smtp_email)
        return True
    except Exception as e:
        print(f"Error when sending email: {e}")
        return False

class UserRegistrationView(APIView):
    renderer_classes = [UserRenderer]

    @swagger_auto_schema(request_body=UserRegistrationSerializer)
    def post(self, request, format=None):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            if 'otp' not in request.data:
                # Generate and send OTP
                email = serializer.validated_data.get('email')
                if not email:
                    return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)
                otp = UserRegistrationSerializer.generate_otp(email)
                name = serializer.validated_data.get('name')
                # Send OTP email
                otp_html = render_to_string('otp_email.html', {'otp': otp, 'name': name, 'date': datetime.date.today()})
                if send_brevo_email(email, name, "Verification OTP for SketchMyHome.AI", otp_html):
                    return Response({'msg': 'OTP sent to your email'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Failed to send OTP'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                # Verify OTP and register user
                user = serializer.save()
                token = get_tokens_for_user(user)
                # Send welcome email
                welcome_html = render_to_string('welcome_email.html', {'name': user.name, 'date': datetime.date.today()})
                if send_brevo_email(user.email, user.name, "Welcome to SketchMyHome.AI", welcome_html):
                    return Response({'token': token, 'msg': 'Registration Successful'}, status=status.HTTP_201_CREATED)
                else:
                    return Response({'token': token, 'msg': 'Registration Successful, but welcome email sending failed'}, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# account/views.py
class UserLoginView(APIView):
    renderer_classes = [UserRenderer]

    @swagger_auto_schema(request_body=UserLoginSerializer)
    def post(self, request, format=None):
        try:
            # Validate request data
            serializer = UserLoginSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            # Authenticate user
            email = serializer.validated_data.get('email')
            password = serializer.validated_data.get('password')
            user = authenticate(email=email, password=password)

            if not user:
                return Response(
                    {'errors': {'non_field_errors': ['Email or Password is not Valid']}},
                    status=status.HTTP_404_NOT_FOUND
                )

            if not user.is_active:
                return Response(
                    {'errors': {'non_field_errors': ['Account is disabled']}},
                    status=status.HTTP_403_FORBIDDEN
                )

            # Generate token with embedded user info
            token = get_tokens_for_user(user)

            # Return successful response
            return Response({
                'token': token,
                'msg': 'Login Success'
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return Response(
                {'errors': {'non_field_errors': ['An error occurred during login']}},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        
class UserProfileView(APIView):
  renderer_classes = [UserRenderer]
  permission_classes = [IsAuthenticated]
  authentication_classes = [JWTAuthentication]
  def get(self, request, format=None):
    serializer = UserProfileSerializer(request.user)
    print(f"Authenticated user: {request.user.id}, {request.user.email}")
    return Response(serializer.data, status=status.HTTP_200_OK)

class UserChangePasswordView(APIView):
  renderer_classes = [UserRenderer]
  authentication_classes = [JWTAuthentication]
  permission_classes = [IsAuthenticated]
  @swagger_auto_schema(request_body=UserChangePasswordSerializer)
  def post(self, request, format=None):
    serializer = UserChangePasswordSerializer(data=request.data, context={'user':request.user})
    serializer.is_valid(raise_exception=True)
    return Response({'msg':'Password Changed Successfully'}, status=status.HTTP_200_OK)

class SendPasswordResetEmailView(APIView):
  renderer_classes = [UserRenderer]
  def post(self, request, format=None):
    serializer = SendPasswordResetEmailSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    email = serializer.validated_data['email']
    user = User.objects.filter(email=email).first()
    if user:
      # Generate password reset token
      uid = urlsafe_base64_encode(force_bytes(user.pk))
      token = default_token_generator.make_token(user)
      reset_link = request.build_absolute_uri(
          reverse('password_reset_confirm', kwargs={'uidb64': uid, 'token': token})
      )
      # Prepare email data
      email_data = {
          'subject': 'Password Reset Request',
          'body': f'Hi {user.username},\n\nUse the link below to reset your password:\n{reset_link}\n\nIf you did not make this request, simply ignore this email.',
          'to_email': email,
      }
      # Send email
      Util.send_email(email_data)
    return Response({'msg':'Password Reset link send. Please check your Email'}, status=status.HTTP_200_OK)

class UserPasswordResetView(APIView):
  renderer_classes = [UserRenderer]
  @swagger_auto_schema(request_body=UserPasswordResetSerializer)
  def post(self, request, uid, token, format=None):
    serializer = UserPasswordResetSerializer(data=request.data, context={'uid':uid, 'token':token})
    serializer.is_valid(raise_exception=True)
    return Response({'msg':'Password Reset Successfully'}, status=status.HTTP_200_OK)
  

class ForgotPasswordView(APIView):
    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            try:
                user = User.objects.get(email=email)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                token = PasswordResetTokenGenerator().make_token(user)
                # Pointing to your React frontend URL
                frontend_url = 'https://sketchmyhome.ai/reset-password/'
                reset_link = f'{frontend_url}{uid}/{token}'
                
                # Email body and sending
                email_body = render_to_string('forgot_password_email.html', {
                    'user': user,
                    'reset_link': reset_link,
                })
                if send_brevo_email(user.email, user.name, "Password Reset for SketchMyHome.ai", email_body):
                    return Response({'success': 'We have sent you a link to reset your password'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Failed to send reset email'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except User.DoesNotExist:
                return Response({'error': 'User with this email does not exist'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ResetPasswordView(APIView):
    def post(self, request, uidb64, token):
        try:
            # Decode the user ID
            user_id = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=user_id)

            # Check if the token is valid
            if not PasswordResetTokenGenerator().check_token(user, token):
                return Response({'error': 'Token is invalid or expired'}, status=status.HTTP_400_BAD_REQUEST)

            # Validate and reset the password
            serializer = ResetPasswordSerializer(data=request.data)
            if serializer.is_valid():
                user.set_password(serializer.validated_data['password'])
                user.save()
                return Response({'success': 'Password reset successful'}, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({'error': 'Token is invalid or expired'}, status=status.HTTP_400_BAD_REQUEST)


class UserLogoutView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    def post(self, request, format=None):
        # Perform logout
        logout(request)
        return Response({'msg': 'Logout Successful'}, status=status.HTTP_200_OK)


class ContactCreateView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ContactSerializer(data=request.data)
        if serializer.is_valid():
            contact = serializer.save()
            # Prepare email content
            to_email = contact.email
            to_name = contact.name
            revert_html = render_to_string('contact_revert_email.html', {'name': to_name,'date':datetime.date.today()})

            # Send email using Brevo email function
            email_sent = send_brevo_email(to_email, to_name, "Thank you for reaching us out!", revert_html)

            if email_sent:
                return Response({'message': 'Contact form submitted and email sent.'}, status=status.HTTP_201_CREATED)
            else:
                return Response({'message': 'Contact form submitted but failed to send email.'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)







