from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from account.serializers import *
from django.contrib.auth import authenticate, logout
from account.renderers import UserRenderer
import boto3,os,datetime
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
  return {
      'refresh': str(refresh),
      'access': str(refresh.access_token),
  }

def send_brevo_email(to_email, to_name, subject, html_content):
    configuration = Configuration()
    configuration.api_key['api-key'] = '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'
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
                email = serializer.validated_data['email']
                otp = UserRegistrationSerializer.generate_otp(email)
                name=serializer.data['name']
                # Send OTP email
                otp_html =render_to_string('otp_email.html', {'otp': otp,'name':name,'date':datetime.date.today()})
                if send_brevo_email(email, serializer.validated_data['name'], "Verification OTP for SketchMyHome.AI", otp_html):
                    return Response({'msg': 'OTP sent to your email'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Failed to send OTP'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                # Verify OTP and register user
                user = serializer.save()
                token = get_tokens_for_user(user)
                # Send welcome email
                welcome_html = render_to_string('welcome_email.html', {'name': user.name,'date':datetime.date.today()})
                if send_brevo_email(user.email, user.name, "Welcome to SketchMyHome.AI", welcome_html):
                    return Response({'token': token, 'msg': 'Registration Successful'}, status=status.HTTP_201_CREATED)
                else:
                    return Response({'token': token, 'msg': 'Registration Successful, but welcome email sending failed'}, status=status.HTTP_201_CREATED)

class UserLoginView(APIView):
  renderer_classes = [UserRenderer]
  @swagger_auto_schema(request_body=UserLoginSerializer)
  def post(self, request, format=None):
    serializer = UserLoginSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    email = serializer.data.get('email')
    password = serializer.data.get('password')
    user = authenticate(email=email, password=password)
    if user is not None:
      token = get_tokens_for_user(user)
      return Response({'token':token, 'msg':'Login Success'}, status=status.HTTP_200_OK)
    else:
      return Response({'errors':{'non_field_errors':['Email or Password is not Valid']}}, status=status.HTTP_404_NOT_FOUND)

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

