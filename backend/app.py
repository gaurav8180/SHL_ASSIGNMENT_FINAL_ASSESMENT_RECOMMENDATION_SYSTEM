import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import from root main to preserve current pipeline location
from main import recommend_assessments

port = int(os.environ.get("PORT", 8000))
app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

class JobRequest(BaseModel):
	job_description: str

@app.get("/")
def read_root():
	return {"message": "SHL Backend is running!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
	return Response(status_code=204)

@app.post("/recommend")
async def recommend(request: JobRequest):
	try:
		recommendations = await recommend_assessments(request.job_description)
		if not recommendations:
			return {
				"recommendations": [],
				"message": "No recommendations found. Please try a more detailed job description."
			}
		return {"recommendations": recommendations}
	except Exception as e:
		return {"recommendations": [], "error": str(e)}

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=port)

import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import recommend_assessments

port = int(os.environ.get("PORT", 8000))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    job_description: str

@app.get("/")
def read_root():
    return {"message": "SHL Backend is running!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/recommend")
async def recommend(request: JobRequest):
    try:
        recs = await recommend_assessments(request.job_description)
        if not recs:
            return {"recommendations": [], "message": "No recommendations found."}
        return {"recommendations": recs}
    except Exception as e:
        return {"recommendations": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
