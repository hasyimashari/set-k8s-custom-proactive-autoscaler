#!/bin/bash

set -e

echo "🚀 test-app deployment..."

kubectl apply -f test-app-ns.yaml

echo "📝 Applying configuration..."
kubectl apply -f test-app-deployment.yaml
kubectl apply -f test-app-service.yaml

echo "⏳ Waiting for test-app deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment.apps/test-app -n test-autoscaler

# Get Minikube IP
MINIKUBE_IP=$(minikube ip)
echo "🌐 Minikube IP: $MINIKUBE_IP"

echo ""
echo "📊 test-app:"
echo "   1. Via NodePort: http://$MINIKUBE_IP:30500"
echo ""