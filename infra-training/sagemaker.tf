terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

provider "null" {}

# Create an S3 Bucket for SageMaker Training & Model Outputs
resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket = "ml-sounds-sandbox-bucket"
  force_destroy = true
}

# IAM Role for SageMaker Execution
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "SageMakerExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

# Attach SageMaker Full Access Policy
resource "aws_iam_policy_attachment" "sagemaker_policy" {
  name       = "sagemaker-policy-attachment"
  roles      = [aws_iam_role.sagemaker_execution_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Custom IAM Policy for SageMaker to Write to S3
resource "aws_iam_policy" "sagemaker_s3_access" {
  name = "SageMakerS3WriteAccess"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ],
        Resource = [
          "arn:aws:s3:::${aws_s3_bucket.sagemaker_bucket.id}",
          "arn:aws:s3:::${aws_s3_bucket.sagemaker_bucket.id}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_access_attach" {
  policy_arn = aws_iam_policy.sagemaker_s3_access.arn
  role       = aws_iam_role.sagemaker_execution_role.name
}

# ✅ Create a SageMaker Training Job using AWS CLI via Terraform
resource "null_resource" "sagemaker_training_job" {
  provisioner "local-exec" {
    command = <<EOT
      aws sagemaker create-training-job \
        --training-job-name "my-training-job" \
        --role-arn "arn:aws:iam::703671899612:role/SageMakerExecutionRole" \
        --algorithm-specification "TrainingImage=703671899612.dkr.ecr.us-east-2.amazonaws.com/my-training-image,TrainingInputMode=File" \
        --resource-config "InstanceType=ml.m5.large,InstanceCount=1,VolumeSizeInGB=10" \
        --stopping-condition "MaxRuntimeInSeconds=3600" \
        --hyper-parameters batch-size=64,epochs=5,lr=0.001 \
        --output-data-config "S3OutputPath=s3://ml-sounds-sandbox-bucket/output/" \
        --region us-east-2
    EOT
  }
}

