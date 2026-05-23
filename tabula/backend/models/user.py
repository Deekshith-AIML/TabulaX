from pydantic import BaseModel, EmailStr, Field
from pydantic_core import core_schema
from typing import Any, Optional
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any, handler: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_obj: core_schema.CoreSchema, handler: Any) -> dict[str, Any]:
        json_schema = handler(core_schema_obj)
        json_schema.update(type="string", example="60d5ecf0e6e4b3b3e4a5f3a0")
        return json_schema

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            PyObjectId: str
        }
        json_schema_extra = {
            "example": {
                "_id": "60d5ecf0e6e4b3b3e4a5f3a0",
                "username": "johndoe",
                "email": "johndoe@example.com",
                "full_name": "John Doe",
                "hashed_password": "hashedpass",
                "is_active": True,
                "created_at": "2023-01-01T12:00:00Z",
                "last_login": None
            }
        }

class User(UserBase):
    id: PyObjectId
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True
        json_encoders = {
            ObjectId: str,
            PyObjectId: str
        }
        json_schema_extra = {
            "example": {
                "id": "60d5ecf0e6e4b3b3e4a5f3a0",
                "username": "johndoe",
                "email": "johndoe@example.com",
                "full_name": "John Doe",
                "is_active": True,
                "created_at": "2023-01-01T12:00:00Z",
                "last_login": None
            }
        } 