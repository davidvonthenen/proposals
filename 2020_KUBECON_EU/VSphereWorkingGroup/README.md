# KubeCon Eu 2020 Demo

The configuration and example used in the [KubeCon EU 2020 - Provider vSphere: All Things vSphere Working Group](https://kccnceu20.sched.com/event/ZevZ/provider-vsphere-all-things-vsphere-working-group-david-vonthenen-vmware) session.

## vCenter Configuration Diagram for RBAC

What the user configuration looks like in this demo:
tenant1@vsphere.local / Myten@nt1
tenant2@vsphere.local / Myten@nt2
eng-ux@vsphere.local / Myt3n@ntux
eng-be@vsphere.local / Myt3n@ntbe

RBAC for each account was setup at the Resource Pool level in the following fashion:

![RBAC in vCenter](https://github.com/dvonthenen/proposals/raw/2020kubeconeupreso/2020_KUBECON_EU/VSphereWorkingGroup/images/users_rbac.png)

## CPI and CSI Deployment in Kubernetes

There are two Kubernetes clusters in the vSphere configuration above. The YAML and scripts provided will deploy CPI and CSI in the following configuration:

![RBAC in vCenter](https://github.com/dvonthenen/proposals/raw/2020kubeconeupreso/2020_KUBECON_EU/VSphereWorkingGroup/images/deployment_demo.png)

## How to Run the Demo

ssh into master in Kubernetes cluster 1:
```bash
# ./tenant1.sh
```

ssh into master in Kubernetes cluster 2:
```bash
# ./tenant2.sh
```
