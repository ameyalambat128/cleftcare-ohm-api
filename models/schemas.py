from pydantic import BaseModel, EmailStr
from typing import List, Optional

# Existing schemas - moved from app.py
class PredictRequest(BaseModel):
    userId: str
    name: str
    communityWorkerName: str
    promptNumber: int
    language: str
    uploadFileName: str
    sendEmail: bool


class GOPRequest(BaseModel):
    userId: str
    name: str
    communityWorkerName: str
    transcript: str
    uploadFileName: str
    sendEmail: bool


class EmailSchema(BaseModel):
    subject: str
    body: str


# New schemas for batch processing
class BatchProcessRequest(BaseModel):
    userId: str
    name: str
    communityWorkerName: str
    sentenceId: int
    transcript: str
    language: str
    uploadFileNames: List[str]  # Multiple audio files for one sentence
    sendEmail: bool
    callbackUrl: Optional[str] = None  # Optional callback URL for async processing


# Standardized response models
class APIResponse(BaseModel):
    success: bool
    data: dict
    metadata: dict


class BatchProcessResponse(BaseModel):
    sentenceId: int
    transcript: str
    gopResults: List[dict]  # All GOP results for each file
    bestGopFile: str
    bestGopScore: float
    ohmRating: float
    processingTime: float
    requestId: str