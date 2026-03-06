name = "Nishant"
age = 30
is_architect = True

print(f"Name: {name}, Age: {age}")

skills = ["C#", ".Net", "Azure", "Python"]
skills.append("LangChain")
print(skills)

person = {"name": "Nishant", "role": "Architect"}
print(person["name"])

def greet(name: str) -> str:
    return f"Hello, {name}"

print(greet("Nishant"))

upper_skills = [s.upper() for s in skills]
print(upper_skills)

class Developer: 
    def __init__(self, name: str, experience: int):
        self.name = name
        self.experience = experience

    def introduce(self) -> str:
        return f"I am {self.name} with {self.experience} years of experience"
    
dev = Developer("Nishant", 8)
print(dev.introduce())