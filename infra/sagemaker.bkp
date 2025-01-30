
# Upload Training Script to S3

resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket = "ml-sounds-sandbox-bucket"
}

resource "aws_s3_object" "training_script" {
  bucket = aws_s3_bucket.sagemaker_bucket.id
  key    = "scripts/train_sagemaker.py"
  source = "train_sagemaker.py"
  acl    = "private"
}

# Create an IAM Role for SageMaker

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

resource "aws_iam_policy_attachment" "sagemaker_policy" {
  name       = "sagemaker-policy-attachment"
  roles      = [aws_iam_role.sagemaker_execution_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Create SageMaker Training Job

resource "aws_sagemaker_training_job" "training_job" {
  name               = "my-training-job"
  role_arn           = aws_iam_role.sagemaker_execution_role.arn
  algorithm_specification {
    training_image     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38"
    training_input_mode = "File"
  }

  resource_config {
    instance_type  = "ml.m5.large"
    instance_count = 1
    volume_size_in_gb = 10
  }

  stopping_condition {
    max_runtime_in_seconds = 3600
  }

  hyperparameters = {
    "batch-size" = "64"
    "epochs"     = "5"
    "lr"         = "0.001"
  }

  input_data_config {
    channel_name = "training" # Folder name in /opt/ml/input/data/
    data_source {
      s3_data_source {
        s3_uri          = "s3://${aws_s3_bucket.sagemaker_bucket.id}/data/"
        s3_data_type    = "S3Prefix"
        s3_data_distribution_type = "FullyReplicated"
      }
    }
  }

  output_data_config {
    s3_output_path = "s3://${aws_s3_bucket.sagemaker_bucket.id}/output/"
  }
}