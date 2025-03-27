# Provider Configuration
provider "aws" {
  region = "us-west-2"
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name        = "sagemaker-training-role"
  description = "Role for SageMaker training jobs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# S3 Access Policy
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
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::ml-sandbox-js-bucket",
          "arn:aws:s3:::ml-sandbox-js-bucket/*"
        ]
      }
    ]
  })
}

# Attach Policies to Role
resource "aws_iam_role_policy_attachment" "s3_access_attach" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.s3_training_data_access.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# SageMaker Training Job with error handling
resource "null_resource" "training_job" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<EOT
      aws sagemaker create-training-job \
      --region us-west-2 \
      --training-job-name "ml-sandbox-js-09" \
      --role-arn ${aws_iam_role.sagemaker_role.arn} \
      --algorithm-specification '{"TrainingImage":"491085419436.dkr.ecr.us-west-2.amazonaws.com/training/ml-sandbox-js:latest","TrainingInputMode":"File"}' \
      --resource-config '{"InstanceType":"ml.m5.large","InstanceCount":1,"VolumeSizeInGB":10}' \
      --input-data-config '[{"ChannelName":"train","DataSource":{"S3DataSource":{"S3Uri":"s3://ml-sandbox-js-bucket/train-data/","S3DataType":"S3Prefix","S3DataDistributionType":"FullyReplicated"}}}]' \
      --output-data-config '{"S3OutputPath":"s3://ml-sandbox-js-bucket/output/"}' \
      --stopping-condition '{"MaxRuntimeInSeconds":420000}'
      EXIT_CODE=$?
      if [ $EXIT_CODE -ne 0 ]; then
        echo "Error creating training job"
        exit $EXIT_CODE
      fi
    EOT
    interpreter = ["/bin/bash", "-c"]
  }
}
        