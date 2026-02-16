
# FAANG++ Enterprise AI Platform

## Added Advanced Components

### RAG Evaluation
- Mean Reciprocal Rank (MRR)
- Recall@K
- RAGAS-style scoring

Run:
python evaluation/rag_evaluation.py

### Distributed Load Testing
- Locust-based stress testing
Run:
locust -f load_testing/locustfile.py --host=http://localhost:8000

### Terraform Infrastructure
- AWS ECS Cluster
- ECR Repository
- CloudWatch Logs

Deploy:
cd terraform
terraform init
terraform apply

This version represents a production-architecture-ready AI platform 
with evaluation, performance testing, and infrastructure automation.
