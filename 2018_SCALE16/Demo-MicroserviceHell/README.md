# SCaLE 16x Demo

## Before we begin...

Requires Helm... Install like so...
https://github.com/kubernetes/helm/blob/master/docs/install.md

Then copy the following repo to your k8s master...
```
git clone git@github.com:dvonthenen/proposals.git
cd proposals/2018_SCALE16/Demo-MicroserviceHell
```

## Deploy Prometheus using helm

helm install stable/prometheus --name metrics --set alertmanager.enabled=false --set alertmanager.persistentVolume.enabled=false --set pushgateway.enabled=false --set kubeStateMetrics.enabled=false --set server.persistentVolume.enabled=false --set nodeExporter.enabled=false --set server.service.type=NodePort -f prometheus.yml

## Deploy Jaeger

helm install incubator/jaeger --name tracing --set cassandra.config.max_heap_size=1024M --set cassandra.config.heap_new_size=256M --set cassandra.resources.requests.memory=2048Mi --set cassandra.resources.requests.cpu=0.4 --set cassandra.resources.limits.memory=2048Mi --set cassandra.resources.limits.cpu=0.4 --set query.service.type=NodePort

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
