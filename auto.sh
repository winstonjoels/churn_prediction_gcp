export PROJECT_ID=mlops-282008
docker build -t gcr.io/${PROJECT_ID}/ml-app:v2 .
gcloud auth configure-docker
docker push gcr.io/${PROJECT_ID}/ml-app:v2
gcloud config set project $PROJECT_ID
gcloud config set compute/zone us-central1
gcloud container clusters create model-cluster --num-nodes=1
kubectl create deployment model-cluster --image=gcr.io/${PROJECT_ID}/ml-app:v2
kubectl expose deployment model-cluster --type=LoadBalancer --port 80 --target-port 8080
kubectl get service
