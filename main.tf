
provider "aws" {
  region = "us-east-1"
}

resource "aws_ecs_cluster" "ai_cluster" {
  name = "enterprise-ai-cluster"
}

resource "aws_ecr_repository" "ai_repo" {
  name = "enterprise-ai-repo"
}

resource "aws_cloudwatch_log_group" "ai_logs" {
  name = "/ecs/enterprise-ai"
}
