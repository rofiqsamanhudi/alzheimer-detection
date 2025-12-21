#!/bin/bash

# Script to push Alzheimer Dashboard to Docker Hub
# Usage: ./push-to-dockerhub.sh <your-dockerhub-username>

if [ -z "$1" ]; then
    echo "Usage: ./push-to-dockerhub.sh <your-dockerhub-username>"
    echo "Example: ./push-to-dockerhub.sh johndoe"
    exit 1
fi

DOCKERHUB_USERNAME=$1
IMAGE_NAME="alzheimer-dashboard"
LOCAL_IMAGE="dashboard-alzheimer-dashboard:latest"

echo "================================================"
echo "Pushing to Docker Hub"
echo "================================================"
echo "Docker Hub Username: $DOCKERHUB_USERNAME"
echo "Image Name: $IMAGE_NAME"
echo ""

# Login to Docker Hub
echo "Step 1: Login to Docker Hub"
sudo docker login

# Tag the image
echo ""
echo "Step 2: Tagging image..."
sudo docker tag $LOCAL_IMAGE $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
sudo docker tag $LOCAL_IMAGE $DOCKERHUB_USERNAME/$IMAGE_NAME:v1.0

# Push the image
echo ""
echo "Step 3: Pushing image to Docker Hub..."
sudo docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
sudo docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:v1.0

echo ""
echo "================================================"
echo "✅ Successfully pushed to Docker Hub!"
echo "================================================"
echo "Image URLs:"
echo "  Latest: $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
echo "  v1.0:   $DOCKERHUB_USERNAME/$IMAGE_NAME:v1.0"
echo ""
echo "To pull this image:"
echo "  docker pull $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
echo ""
