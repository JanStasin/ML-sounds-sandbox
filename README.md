# ML-sounds-sandbox

// TODO



## Getting Started

### Prerequisites
- Python 3.8 or higher
- Pip
- PyTorch
- FastAPI
- Docker (optional, for containerization)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd ML-sounds-sandbox
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Steps to Run the Project

Run the FastAPI app:
```bash
uvicorn app:app --reload
```
Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t ml-sounds-sandbox .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 ML-sounds-sandbox
   ```

3. Access the API at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Infra

- Run `terraform plan` to view plan of Terraform.
- Run `terraform apply` to deploy the infrastructure.
- Run `terraform destroy` to stroy the infrastructure in AWS.


### 
Update Docker Image

```
docker build -t ml-sounds-sandbox .
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 703671899612.dkr.ecr.us-east-2.amazonaws.com
docker tag ml-sounds-sandbox:latest 703671899612.dkr.ecr.us-east-2.amazonaws.com/ml-sounds-sandbox:latest
docker push 703671899612.dkr.ecr.us-east-2.amazonaws.com/ml-sounds-sandbox:latest
```