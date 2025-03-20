# Provider Configuration
provider "aws" {
  region = "us-west-2"
}

# Use AmazonSageMakerFullAccess policy which includes necessary permissions
resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-training-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  # Add explicit delay for AWS service propagation
  provisioner "local-exec" {
    command = "sleep 30"
  }
}

# Add the new S3 permissions here
resource "aws_iam_policy" "s3_training_data_access" {
  name        = "s3-training-data-access"
  description = "Allows SageMaker to access training data in S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::ml-sandbox-js-bucket",
          "arn:aws:s3:::ml-sandbox-js-bucket/*"
        ]
      }
    ]
  })
  provisioner "local-exec" {
    command = "sleep 30"
  }
}

# Attach the policy to your existing role
resource "aws_iam_role_policy_attachment" "s3_access_attach" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.s3_training_data_access.arn
}

# Attach comprehensive managed policy
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# CloudWatch Logs Configuration
resource "aws_cloudwatch_log_group" "debug_logs" {
  name              = "/aws/debug-ml-sandbox-js"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_stream" "app_stream" {
  name           = "application"
  log_group_name = aws_cloudwatch_log_group.debug_logs.name
}

resource "null_resource" "terraform_apply" {
  provisioner "local-exec" {
    command = <<EOT
      set -euo pipefail
      terraform init -input=false
      terraform apply -auto-approve -input=false 2>&1 | tee terraform.log
      EXIT_CODE=$?
      if [ $EXIT_CODE -ne 0 ]; then
        echo "Detailed error information:"
        tail -n 20 terraform.log
        exit $EXIT_CODE
      fi
    EOT
    interpreter = ["/bin/bash", "-c"]
  }
}

# SageMaker Training Job
resource "null_resource" "sagemaker_training_job" {
  provisioner "local-exec" {
    command = <<EOT
aws sagemaker create-training-job \
--region us-west-2 \
--training-job-name "ml-sandbox-js-3" \
--role-arn ${aws_iam_role.sagemaker_role.arn} \
--algorithm-specification '{"TrainingImage":"491085419436.dkr.ecr.us-west-2.amazonaws.com/training/ml-sandbox-js:latest","TrainingInputMode":"File"}' \
--resource-config '{"InstanceType":"ml.m5.large","InstanceCount":1,"VolumeSizeInGB":10}' \
--input-data-config '[{"ChannelName":"train","DataSource":{"S3DataSource":{"S3Uri":"s3://ml-sandbox-js-bucket/train-data/","S3DataType":"S3Prefix","S3DataDistributionType":"FullyReplicated"}}}]' \
--output-data-config '{"S3OutputPath":"s3://ml-sandbox-js-bucket/output/"}' \
--stopping-condition '{"MaxRuntimeInSeconds":36000}'
EOT
  }
}