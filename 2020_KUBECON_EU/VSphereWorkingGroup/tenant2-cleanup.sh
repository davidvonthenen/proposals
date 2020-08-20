#!/bin/bash

set -o pipefail
set -o xtrace


cd tenant2-be/csi
kubectl delete -f vsphere-csi-node-ds.yaml
kubectl delete -f vsphere-csi-controller-deployment.yaml
kubectl delete -f vsphere-csi-controller-rbac.yaml
kubectl delete secret vsphere-config-secret-be --namespace=kube-system
cd ../..

cd tenant2-ux/csi
kubectl delete -f vsphere-csi-node-ds.yaml
kubectl delete -f vsphere-csi-controller-deployment.yaml
kubectl delete -f vsphere-csi-controller-rbac.yaml
kubectl delete secret vsphere-config-secret-ux --namespace=kube-system
cd ../..

cd tenant2/cpi
kubectl delete -f vsphere-cloud-controller-manager-ds.yaml
kubectl delete -f cloud-controller-manager-role-bindings.yaml
kubectl delete -f cloud-controller-manager-roles.yaml
kubectl delete configmap cloud-config --namespace=kube-system
cd ../..
