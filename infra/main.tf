# -----------------------------------------------------------------------------
# Property valuation service — minimal AWS infrastructure
#
# PROVISIONS
#   - S3 bucket for model artifacts and data (versioned, encrypted, private).
#     Point PVS_MLFLOW_URI / MLflow's artifact root at s3://<bucket>/mlruns.
#   - ECR repository for the service image (scan on push).
#   - ECS Fargate cluster, task definition and service running the API on 8000.
#   - CloudWatch log group for the container (prediction logs land here).
#   - Task execution role (pull image, write logs) and a task role that can
#     read/write only this bucket.
#
# DELIBERATELY OMITTED (out of scope for a portfolio project, required for prod)
#   - Networking: VPC, private subnets, NAT, security-group hardening and a
#     load balancer / TLS. Subnets and a security group are passed in as vars.
#   - IAM hardening: permission boundaries, SCPs, KMS customer-managed keys.
#   - A managed MLflow backend store (RDS) — the SQLite store does not scale
#     beyond one writer.
#   - Autoscaling policies, WAF, alarms wiring to PagerDuty/Slack.
#   - Remote Terraform state (S3 + DynamoDB lock).
# -----------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "ap-southeast-2" # Sydney
}

variable "project" {
  type    = string
  default = "property-valuation"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "subnet_ids" {
  description = "Subnets for the Fargate service (networking is out of scope here)."
  type        = list(string)
}

variable "security_group_ids" {
  type = list(string)
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------- storage
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project}-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled" # every model artifact version is recoverable -> rollback
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------- image registry
resource "aws_ecr_repository" "api" {
  name                 = var.project
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# ---------------------------------------------------------------- IAM
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.project}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name               = "${var.project}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "bucket_access" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "task_bucket" {
  name   = "artifact-bucket"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.bucket_access.json
}

# ---------------------------------------------------------------- container service
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project}"
  retention_in_days = 30
}

resource "aws_ecs_cluster" "main" {
  name = var.project
}

resource "aws_ecs_task_definition" "api" {
  family                   = var.project
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name         = "api"
    image        = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "PVS_ARTIFACT_BUCKET", value = aws_s3_bucket.artifacts.bucket },
    ]
    healthCheck = {
      command  = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]
      interval = 30
      timeout  = 5
      retries  = 3
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${var.project}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = false
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true # a task that fails /health rolls the deployment back automatically
  }
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}
