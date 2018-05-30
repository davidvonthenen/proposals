# Open Source Summit Japan Demo
## Application Monitoring and Tracing in Kubernetes: Avoiding Microservice Hell!

## Before we begin...

Requires Helm... Install like so...  
https://github.com/kubernetes/helm/blob/master/docs/install.md

Then copy the following repo to your k8s master...
```
git clone git@github.com:dvonthenen/proposals.git
cd proposals/2018_OSS_JAPAN/Demo-MicroserviceHell
```

You can find the Go source code, Dockerfile, etc for the backend service here:  
https://github.com/dvonthenen/jop-stack/tree/master/backend

You can find the Go source code, Dockerfile, etc for the frontend service here:  
https://github.com/dvonthenen/jop-stack/tree/master/frontend

The Docker images are published here:  
https://hub.docker.com/r/dvonthenen/jop-backend/  
https://hub.docker.com/r/dvonthenen/jop-frontend/

## Deploy Prometheus using helm

helm install stable/prometheus --name metrics --version 5.4.1 --set alertmanager.enabled=false --set alertmanager.persistentVolume.enabled=false --set pushgateway.enabled=false --set kubeStateMetrics.enabled=false --set server.persistentVolume.enabled=false --set nodeExporter.enabled=false --set server.service.type=NodePort -f prometheus.yml

## Deploy Jaeger

helm install incubator/jaeger --name tracing --version 0.2.4 --set cassandra.config.max_heap_size=1024M --set cassandra.config.heap_new_size=256M --set cassandra.resources.requests.memory=2048Mi --set cassandra.resources.requests.cpu=0.4 --set cassandra.resources.limits.memory=2048Mi --set cassandra.resources.limits.cpu=0.4

## Warning!!

Since we are using NodePort, dont forget to change jaeger and prometheus UI firewall forwarding!

## Deploy the Sample App

```
cd services
kubectl create -f backend.yaml
kubectl create -f frontend.yaml
cd ..

Dont forget to change frontend firewall forwarding

cd deployments
kubectl create -f backend.yaml
kubectl create -f frontend.yaml
cd ..
```

## Run a simple curl on the App

```
curl http://<IP>:<New Port>
```

## Clean Up

```
kubectl delete service frontend
kubectl delete service backend
kubectl delete deployment frontend
kubectl delete deployment backend

helm delete --purge metrics
helm delete --purge tracing
```
