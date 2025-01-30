## Pipeline

1️⃣ VPC Module
Creates 1 VPC with 2 public and 2 private subnets.
Enables NAT Gateway for internet access.
2️⃣ ECS Module
Creates an ECS Cluster (my-ecs-cluster).
Deploys an ECS Fargate service in a private subnet.
3️⃣ ALB Module
Creates an Application Load Balancer (my-app-lb).
Adds a target group for the ECS service.
Creates an ALB Security Group to allow HTTP traffic.
4️⃣ ECS Task Execution Role
Uses an IAM Role to allow ECS to pull images from ECR.
5️⃣ Security Groups
ALB Security Group → Allows HTTP traffic.
ECS Security Group → Allows traffic from ALB.