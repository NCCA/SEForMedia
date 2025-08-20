#!/usr/bin/env -S uv run --script


class Person:
    def __init__(self, name):
        self._name = name

    def introduce(self):
        return f"I am {self._name}."


class Employee(Person):
    def __init__(self, name, employee_id):
        super().__init__(name)
        self.employee_id = employee_id

    def introduce(self):
        return f"I am {self._name} and my employee ID is {self.employee_id}."


person = Person("Jon")
employee = Employee("Jon", "E123")

print(person.introduce())
print(employee.introduce())
