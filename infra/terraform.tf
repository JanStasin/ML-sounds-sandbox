provider "aws" {
  region = "us-east-2"
}

# --- VPC Setup ---
resource "aws_vpc" "app_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "App-VPC"
  }
}

resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.app_vpc.id
  cidr_block              = "10.0.0.0/24"
  availability_zone       = "us-east-2a" # Fixed AZ
  map_public_ip_on_launch = true
  tags = {
    Name = "Public-Subnet"
  }
}

resource "aws_subnet" "private_subnet" {
  vpc_id            = aws_vpc.app_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-2a" # Fixed AZ
  tags = {
    Name = "Private-Subnet"
  }
}

resource "aws_internet_gateway" "app_igw" {
  vpc_id = aws_vpc.app_vpc.id
  tags = {
    Name = "App-IGW"
  }
}

resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.app_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.app_igw.id
  }

  tags = {
    Name = "Public-Route-Table"
  }
}

resource "aws_route_table_association" "public_subnet_association" {
  subnet_id       = aws_subnet.public_subnet.id
  route_table_id  = aws_route_table.public_route_table.id
}

resource "aws_route_table" "private_route_table" {
  vpc_id = aws_vpc.app_vpc.id
  tags = {
    Name = "Private-Route-Table"
  }
}

resource "aws_route_table_association" "private_subnet_association" {
  subnet_id       = aws_subnet.private_subnet.id
  route_table_id  = aws_route_table.private_route_table.id
}

# --- Security Groups ---
resource "aws_security_group" "ecs_security_group" {
  vpc_id = aws_vpc.app_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ECS-SG"
  }
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecsTaskExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "ECS Task Execution Role"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- VPC Endpoints for ECR ---
resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id            = aws_vpc.app_vpc.id
  service_name      = "com.amazonaws.us-east-2.ecr.dkr"
  private_dns_enabled = true
  subnet_ids        = [aws_subnet.private_subnet.id] # Single AZ
  security_group_ids = [aws_security_group.ecs_security_group.id]
  vpc_endpoint_type = "Interface"

  tags = {
    Name = "ECR-DKR-Endpoint"
  }
}

resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id            = aws_vpc.app_vpc.id
  service_name      = "com.amazonaws.us-east-2.ecr.api"
  private_dns_enabled = true
  subnet_ids        = [aws_subnet.private_subnet.id] # Single AZ
  security_group_ids = [aws_security_group.ecs_security_group.id]
  vpc_endpoint_type = "Interface"

  tags = {
    Name = "ECR-API-Endpoint"
  }
}

resource "aws_vpc_endpoint" "logs" {
  vpc_id            = aws_vpc.app_vpc.id
  service_name      = "com.amazonaws.us-east-2.logs"
  private_dns_enabled = true
  subnet_ids        = [aws_subnet.private_subnet.id] # Single AZ
  security_group_ids = [aws_security_group.ecs_security_group.id]
  vpc_endpoint_type = "Interface"

  tags = {
    Name = "CloudWatch-Logs-Endpoint"
  }
}

# --- ECS Cluster ---
resource "aws_ecs_cluster" "app_cluster" {
  name = "App-Cluster"
}

resource "aws_ecs_task_definition" "app_task" {
  family                   = "app-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "app-container"
      image     = "703671899612.dkr.ecr.us-east-2.amazonaws.com/ml-sounds-sandbox:latest"
      cpu       = 256
      memory    = 512
      essential = true
      portMappings = [
        {
          containerPort = 80
          hostPort      = 80
        }
      ]
    }
  ])
}

resource "aws_ecs_service" "app_service" {
  name            = "app-service"
  cluster         = aws_ecs_cluster.app_cluster.id
  task_definition = aws_ecs_task_definition.app_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [aws_subnet.private_subnet.id]
    security_groups = [aws_security_group.ecs_security_group.id]
  }
}