from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

@app.get("/test")  # Correct decorator for GET requests
def hello():
    return {"message": "hello world2"}  # Return a JSON response