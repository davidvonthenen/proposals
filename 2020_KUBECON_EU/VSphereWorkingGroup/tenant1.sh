#!/bin/bash

set -o pipefail
set -o xtrace

kubectl taint nodes k8smaster.local node.cloudprovider.kubernetes.io/uninitialized=true:NoSchedule
kubectl taint nodes k8sworker1.local node.cloudprovider.kubernetes.io/uninitialized=true:NoSchedule
kubectl taint nodes k8sworker2.local node.cloudprovider.kubernetes.io/uninitialized=true:NoSchedule

cd tenant1/cpi
kubectl create configmap cloud-config --from-file=vsphere.conf --namespace=kube-system
kubectl create -f cloud-controller-manager-role-bindings.yaml
kubectl create -f cloud-controller-manager-roles.yaml
kubectl create -f vsphere-cloud-controller-manager-ds.yaml
cd ..

cd csi
kubectl create secret generic vsphere-config-secret --from-file=csi-vsphere.conf --namespace=kube-system
kubectl create -f vsphere-csi-controller-rbac.yaml
kubectl create -f vsphere-csi-controller-deployment.yaml
kubectl create -f vsphere-csi-node-ds.yaml
cd ../..
