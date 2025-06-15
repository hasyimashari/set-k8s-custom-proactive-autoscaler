#!/bin/bash

# Cleanup Script for Prometheus + Grafana Setup
# This script removes all Prometheus and Grafana resources from Minikube

set -e

echo "🧹 Cleaning up Prometheus + Grafana from Minikube..."

# First, let's see what kube-state-metrics resources exist
echo ""
echo "🔍 Checking existing kube-state-metrics resources..."
echo "   Checking in kube-system namespace:"
kubectl get serviceaccount,deployment,service -n kube-system | grep kube-state-metrics || echo "   No kube-state-metrics resources found in kube-system"
echo ""
echo "   Checking cluster-wide RBAC:"
kubectl get clusterrole,clusterrolebinding | grep kube-state-metrics || echo "   No kube-state-metrics RBAC resources found"
echo ""

# Function to check if resource exists before deleting
safe_delete() {
    local resource_type=$1
    local resource_name=$2
    local namespace=${3:-""}
    
    if [ -n "$namespace" ]; then
        if kubectl get $resource_type $resource_name -n $namespace >/dev/null 2>&1; then
            echo "   ✓ Found and deleting $resource_type/$resource_name in namespace $namespace"
            kubectl delete $resource_type $resource_name -n $namespace
        else
            echo "   ⚠️  $resource_type/$resource_name not found in namespace $namespace"
        fi
    else
        if kubectl get $resource_type $resource_name >/dev/null 2>&1; then
            echo "   ✓ Found and deleting $resource_type/$resource_name"
            kubectl delete $resource_type $resource_name
        else
            echo "   ⚠️  $resource_type/$resource_name not found (cluster-wide)"
        fi
    fi
}

# Delete Grafana resources
echo "🗑️  Removing Grafana resources..."
safe_delete "service" "grafana" "monitoring"
safe_delete "deployment" "grafana" "monitoring"
safe_delete "configmap" "grafana-datasources" "monitoring"

# Delete node-exporter resources
echo "🗑️  Removing node-exporter resources..."
safe_delete "service" "node-exporter" "monitoring"
safe_delete "daemonset" "node-exporter" "monitoring"

# Delete kube-state-metrics resources
echo "🗑️  Removing kube-state-metrics resources..."
safe_delete "service" "kube-state-metrics" "kube-system"
safe_delete "deployment" "kube-state-metrics" "kube-system"
safe_delete "serviceaccount" "kube-state-metrics" "kube-system"
safe_delete "clusterrole" "kube-state-metrics"
safe_delete "clusterrolebinding" "kube-state-metrics"

# Delete Prometheus resources
echo "🗑️  Removing Prometheus resources..."
safe_delete "service" "prometheus-service" "monitoring"
safe_delete "deployment" "prometheus-deployment" "monitoring"
safe_delete "configmap" "prometheus-server-conf" "monitoring"

# Delete RBAC resources
echo "🗑️  Removing RBAC resources..."
safe_delete "clusterrolebinding" "prometheus"
safe_delete "clusterrole" "prometheus"

# Delete monitoring namespace (this will also clean up any remaining resources)
echo "🗑️  Removing monitoring namespace..."
if kubectl get namespace monitoring >/dev/null 2>&1; then
    echo "   Deleting namespace monitoring (this may take a moment...)"
    kubectl delete namespace monitoring --timeout=60s
fi

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "🔍 Verification commands:"
echo "   - Check remaining pods: kubectl get pods -n monitoring"
echo "   - Check remaining services: kubectl get services -n monitoring"
echo "   - Check namespaces: kubectl get namespaces"
echo ""
echo "📝 Note: The metrics-server addon was left enabled."
echo "   To disable it, run: minikube addons disable metrics-server"
echo ""
echo "🚀 To redeploy, run the setup script again."