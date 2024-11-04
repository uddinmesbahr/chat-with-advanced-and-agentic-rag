from fastapi import FastAPI
from src.graph import WorkFlow

# Initialize FastAPI app
app = FastAPI()

# Initialize your workflow
workflow = WorkFlow()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Workflow!"}

@app.post("/invoke")
def invoke_workflow(payload: dict):
    # Assuming 'invoke' is a method of WorkFlow that processes the input
    result = workflow.app.invoke(payload)
    return {"result": result}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
