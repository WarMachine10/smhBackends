from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.core.exceptions import PermissionDenied
from .models import Project, SubProject
from .serializers import CreateFloorplanningProjectSerializer, SubProjectSerializer, ProjectWithSubProjectsSerializer
from .permissions import IsOwner
from Subscriptions.models import Subscription, CustomerProfile

class ProjectListCreateView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        projects = Project.objects.filter(user=request.user).prefetch_related(
            'subprojects',
            'subprojects__floorplanning',
            'subprojects__floorplanning__user_files'
        )

        # Serialize the projects with their related subprojects and floorplanning data
        serializer = ProjectWithSubProjectsSerializer(projects, many=True)
        return Response(serializer.data)

    def post(self, request):
        try:
            user_profile = CustomerProfile.objects.get(user=request.user)
        except CustomerProfile.DoesNotExist:
            return Response({'detail': 'User profile not found.'}, status=status.HTTP_404_NOT_FOUND)

        subscription = Subscription.objects.filter(
            user=request.user,
            status='ACTIVE',
            end_date__gte=timezone.now()
        ).first()

        if not subscription:
            if user_profile.has_free_project:
                return Response({'detail': 'You have already created a free project. Please subscribe to create more projects.'}, status=status.HTTP_403_FORBIDDEN)
            else:
                user_profile.has_free_project = True
                user_profile.save()

        data = request.data.copy()
        data['user'] = request.user.id

        serializer = ProjectWithSubProjectsSerializer(data=data)

        if serializer.is_valid():
            project = serializer.save()

            # Update project count
            user_profile.project_count += 1
            user_profile.save()

            # Reload the project with its relations to ensure full data for response
            project_with_relations = Project.objects.filter(id=project.id).prefetch_related(
                'subprojects',
                'subprojects__floorplanning',
                'subprojects__floorplanning__user_files'
            ).first()

            response_serializer = ProjectWithSubProjectsSerializer(project_with_relations)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)

        return Response({
            'detail': 'Validation failed for project creation.',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


# projects/views.py

class SubProjectRetrieveUpdateDestroyView(APIView):
    """
    API view for retrieving, updating, and deleting individual subprojects.
    Ensures users can only access subprojects of projects they own.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        # Get subproject and verify ownership
        subproject = get_object_or_404(SubProject, pk=pk)
        if subproject.project.user != self.request.user:
            self.permission_denied(
                self.request,
                message="You do not have permission to access this subproject."
            )
        return subproject

    def get(self, request, pk):
        subproject = self.get_object(pk)
        serializer = SubProjectSerializer(subproject)
        return Response(serializer.data)

    def put(self, request, pk):
        subproject = self.get_object(pk)
        serializer = SubProjectSerializer(subproject, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        subproject = self.get_object(pk)
        user_profile = CustomerProfile.objects.get(user=request.user)
        user_profile.project_count -= 1
        user_profile.save()
        subproject.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# projects/views.py

class SubProjectListCreateView(APIView):
    """
    API view for listing and creating subprojects.
    Ensures users can only create subprojects for projects they own.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_project(self, project_id, user):
        """
        Helper function to get the project for the authenticated user.
        Returns a project if the user owns it, else raises a Project.DoesNotExist error.
        """
        try:
            return Project.objects.get(pk=project_id, user=user)
        except Project.DoesNotExist:
            raise Project.DoesNotExist(f"You do not have permission to access this project.")

    def get(self, request, project_id):
        """
        List subprojects for a specific project owned by the authenticated user.
        """
        # Ensure that the user is the owner of the project
        try:
            project = self.get_project(project_id, request.user)
        except Project.DoesNotExist as e:
            return Response({"detail": str(e)}, status=status.HTTP_403_FORBIDDEN)

        # Filter subprojects for the project
        subprojects = SubProject.objects.filter(project=project).prefetch_related(
            'floorplanning',
            'floorplanning__user_files'
        )
        serializer = SubProjectSerializer(subprojects, many=True)
        return Response(serializer.data)

    def post(self, request, project_id):
        """
        Create a subproject for a specific project owned by the authenticated user.
        """
        # Verify the user owns the parent project
        try:
            project = self.get_project(project_id, request.user)
        except Project.DoesNotExist as e:
            return Response({"detail": str(e)}, status=status.HTTP_403_FORBIDDEN)

        # Add the project_id to the request data before saving the subproject
        request.data['project'] = project.id

        # Serialize and save the subproject
        serializer = SubProjectSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



# projects/views.py

class ProjectRetrieveUpdateDestroyView(APIView):
    """
    API view for retrieving, updating, and deleting individual projects.
    Ensures users can only access their own projects.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated, IsOwner]

    def get_object(self, pk):
        # Get project and check permissions
        project = get_object_or_404(Project, pk=pk)
        self.check_object_permissions(self.request, project)
        return project

    def get(self, request, pk):
        project = self.get_object(pk)
        serializer = ProjectWithSubProjectsSerializer(project)
        return Response(serializer.data)

    def put(self, request, pk):
        project = self.get_object(pk)
        data = request.data.copy()
        data['user'] = request.user.id

        serializer = ProjectWithSubProjectsSerializer(project, data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        project = self.get_object(pk)
        user_profile = CustomerProfile.objects.get(user=request.user)
        user_profile.project_count -= 1
        user_profile.save()
        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)