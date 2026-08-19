"""Pydantic request/response models for auth."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    name: str = Field(..., min_length=1, max_length=128)
    email: EmailStr
    # Frontend pre-fills "personal", so this always has a value.
    institute: str = Field(default="personal", min_length=1, max_length=255)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class ProfileUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    email: EmailStr
    institute: str = Field(default="personal", min_length=1, max_length=255)


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    name: str
    institute: str | None = None
    is_verified: bool
    created_at: str | None = None
    last_login: str | None = None


class AuthResponse(BaseModel):
    user: UserOut
