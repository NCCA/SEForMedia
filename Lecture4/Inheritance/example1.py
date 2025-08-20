#!/usr/bin/env -S uv run --script
# Parent Class


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, my name is {self.name} and I am {self.age} years old."


# Child Class (inherits from Person)
class Employee(Person):
    def __init__(self, name, age, employee_id):
        # Call the parent class constructor
        super().__init__(name, age)
        self.employee_id = employee_id

    def working(self):
        return f"{self.name} is working."


# Using the classes
person = Person("Alice", 30)
employee = Employee("Bob", 25, "E123")

print(person.introduce())
print(employee.introduce())
print(employee.working())

print(person.__dir__())
print(employee.__dir__())
