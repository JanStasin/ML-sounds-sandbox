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
  name     = "my-training-job"
  role_arn = aws_iam_role.sagemaker_execution_role.arn

  algorithm_specification {
    training_image     = "IMAGE_NAME"  # Replace with your ECR image
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

  output_data_config {
    s3_output_path = "s3://${aws_s3_bucket.sagemaker_bucket.id}/output/"
  }
}
