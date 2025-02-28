
provider "aws" {
  region = "us-east-2"
}

provider "null" {}

resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket = "ml-sounds-sandbox-bucket"
}


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
}

resource "aws_iam_policy" "sagemaker_policy" {
  name        = "sagemaker-training-policy"
  description = "Policy for SageMaker to access S3 and CloudWatch"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::ml-sounds-sandbox-bucket",
          "arn:aws:s3:::ml-sounds-sandbox-bucket/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_policy" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}


resource "aws_iam_policy" "sagemaker_s3_policy" {
  name        = "sagemaker-s3-access"
  description = "Allows SageMaker to access S3 bucket for training data"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = [
          "arn:aws:s3:::ml-sounds-sandbox-bucket",
          "arn:aws:s3:::ml-sounds-sandbox-bucket/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_sagemaker_s3" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}


resource "null_resource" "sagemaker_training_job" {
  provisioner "local-exec" {
    command = <<EOT
      aws sagemaker create-training-job \
        --region us-east-2 \
        --training-job-name "ml-sounds-sandbox" \
        --role-arn "arn:aws:iam::703671899612:role/sagemaker-training-role" \
        --algorithm-specification '{"TrainingImage":"703671899612.dkr.ecr.us-east-2.amazonaws.com/xxxxxxxxxx:latest","TrainingInputMode":"File"}' \
        --resource-config '{"InstanceType":"ml.m5.large","InstanceCount":1,"VolumeSizeInGB":10}' \
        --input-data-config '[{"ChannelName":"train","DataSource":{"S3DataSource":{"S3Uri":"s3://ml-sounds-sandbox-bucket/train/","S3DataType":"S3Prefix","S3DataDistributionType":"FullyReplicated"}}}]' \
        --output-data-config '{"S3OutputPath":"s3://ml-sounds-sandbox-bucket/output/"}' \
        --stopping-condition '{"MaxRuntimeInSeconds":3600}' \
        --hyperparameters "n-estimators=200,max-depth=10"
    EOT
  }
}



resource "null_resource" "trigger_sagemaker_job" {
  provisioner "local-exec" {
    command = <<EOT
      aws sagemaker create-training-job \
        --training-job-name "sagemaker-training-cli-job" \
        --algorithm-specification TrainingImage=703671899612.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest,TrainingInputMode=File \
        --role-arn ${aws_iam_role.sagemaker_role.arn} \
        --resource-config InstanceType=ml.m5.large,InstanceCount=1,VolumeSizeInGB=10 \
        --input-data-config "[{\"ChannelName\":\"training\",\"DataSource\":{\"S3DataSource\":{\"S3Uri\":\"s3://ml-sounds-sandbox-bucket/train/\",\"S3DataType\":\"S3Prefix\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]" \
        --output-data-config S3OutputPath=s3://ml-sounds-sandbox-bucket/output/ \
        --stopping-condition MaxRuntimeInSeconds=3600
    EOT
  }
}
