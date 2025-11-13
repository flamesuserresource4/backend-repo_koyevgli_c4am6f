"""
Database Schemas for Pixels Mind

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class User(BaseModel):
    email: Optional[str] = Field(None, description="User email")
    name: Optional[str] = Field(None, description="Display name")
    auth_provider: Literal["guest", "email"] = "guest"
    avatar_url: Optional[str] = None
    is_active: bool = True


class Upload(BaseModel):
    user_id: Optional[str] = None
    kind: Literal["model", "apparel"]
    filename: str
    url: str


class Job(BaseModel):
    user_id: Optional[str] = None
    preset_id: Optional[str] = None
    aspect_ratio: Literal["portrait", "square", "landscape", "stories"] = "square"
    num_images: int = Field(1, ge=1, le=8)
    packs: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    error: Optional[str] = None


class Asset(BaseModel):
    user_id: Optional[str] = None
    job_id: str
    url: str
    kind: Literal["generated", "upload"] = "generated"
    meta: Dict[str, Any] = Field(default_factory=dict)


class Preset(BaseModel):
    id: str
    title: str
    description: str
    lighting: str
    background: str
    pose: str
    accent: Optional[str] = None
