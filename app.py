import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import recommend_assessments

# Use PORT from environment or default to 8000
port = int(os.environ.get("PORT", 8000))
app = FastAPI()

# Configure CORS for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    job_description: str

@app.get("/")
def read_root():
    return {"message": "SHL Backend is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/recommend")
async def recommend(request: JobRequest):
    """
    Assessment Recommendation Endpoint
    """
    try:
        recommendations = await recommend_assessments(request.job_description)
        
        if not recommendations:
            return {
                "recommendations": [],
                "message": "No recommendations found. Please try a more detailed job description."
            }
        
        return {"recommendations": recommendations}
    except Exception as e:
        return {
            "recommendations": [],
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)