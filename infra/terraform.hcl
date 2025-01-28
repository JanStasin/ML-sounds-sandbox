provider "aws" {
  region = "us-east-1"
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

resource "aws_subnet" "public_subnets" {
  count                   = 2
  vpc_id                  = aws_vpc.app_vpc.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = {
    Name = "Public-Subnet-${count.index + 1}"
  }
}

resource "aws_subnet" "private_subnets" {
  count             = 2
  vpc_id            = aws_vpc.app_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = {
    Name = "Private-Subnet-${count.index + 1}"
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
  count           = length(aws_subnet.public_subnets)
  subnet_id       = aws_subnet.public_subnets[count.index].id
  route_table_id  = aws_route_table.public_route_table.id
}

resource "aws_route_table" "private_route_table" {
  vpc_id = aws_vpc.app_vpc.id
  tags = {
    Name = "Private-Route-Table"
  }
}

resource "aws_route_table_association" "private_subnet_association" {
  count           = length(aws_subnet.private_subnets)
  subnet_id       = aws_subnet.private_subnets[count.index].id
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

# --- VPC Endpoints for ECR ---
resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id            = aws_vpc.app_vpc.id
  service_name      = "com.amazonaws.${var.aws_region}.ecr.dkr"
  private_dns_enabled = true
  subnet_ids        = [aws_subnet.private_subnets[0].id] # Single AZ
  security_group_ids = [aws_security_group.ecs_security_group.id]

  tags = {
    Name = "ECR-DKR-Endpoint"
  }
}

resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id            = aws_vpc.app_vpc.id
  service_name      = "com.amazonaws.${var.aws_region}.ecr.api"
  private_dns_enabled = true
  subnet_ids        = [aws_subnet.private_subnets[0].id] # Single AZ
  security_group_ids = [aws_security_group.ecs_security_group.id]

  tags = {
    Name = "ECR-API-Endpoint"
  }
}

resource "aws_vpc_endpoint" "logs" {
  vpc_id            = aws_vpc.app_vpc.id
  service_name      = "com.amazonaws.${var.aws_region}.logs"
  private_dns_enabled = true
  subnet_ids        = [aws_subnet.private_subnets[0].id] # Single AZ
  security_group_ids = [aws_security_group.ecs_security_group.id]

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
      image     = "703671899612.dkr.ecr.us-east-2.amazonaws.com/ML-sounds-sandbox:latest"
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
    subnets         = aws_subnet.private_subnets[*].id
    security_groups = [aws_security_group.ecs_security_group.id]
  }
}
