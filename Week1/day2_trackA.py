from pydantic import BaseModel, EmailStr, Field, ValidationError
from typing import Optional
from datetime import date

class Address(BaseModel):
    street: str
    city: str
    pincode: str

class Customer(BaseModel):
    id: int
    name: str = Field(min_length=2, max_length=50)
    email: str
    age: int = Field(ge=10, le=100)
    is_active: bool = True
    address: Optional[Address] = None

# Create a valid customer
customer = Customer(
    id=1,
    name="Nishant",
    email="nishant@abc.com",
    age=30,
    address=Address(street="MG Road", city="Bangalore", pincode="560001")
)

print(customer)
print(customer.model_dump())
print(customer.model_dump_json())

try:
    bad_customer = Customer(
        id=2,
        name="X",
        email="not-an-email",
        age=15
    )
except ValidationError as e:
    print("\nValidation Errors:")
    print(e)

class AskRequest(BaseModel):
    question: str = Field(min_length=5, description="The question to ask the AI")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class AskResponse(BaseModel):
    answer: str
    tokens_used: int

request = AskRequest(question="What is LangChain?", temperature=0.5)
print(f"\nRequest: {request}")
print(f"As dict: {request.model_dump()}")