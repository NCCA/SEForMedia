#!/usr/bin/env -S uv run --script


class Person:
    def introduce(self):
        return "I am a person."


class Employee(Person):
    def introduce(self):
        return "I am an employee."


person = Person()
employee = Employee()

print(person.introduce())
print(employee.introduce())
