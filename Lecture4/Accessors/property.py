#!/usr/bin/env -S uv run --script


class Person:
    def __init__(self, name):
        self._name = name  # The underscore signifies a "protected" variable

    # Define the getter method using @property
    @property
    def name(self):
        return self._name

    # Define the setter method using @property_name.setter
    @name.setter
    def name(self, name):
        if isinstance(name, str) and len(name) > 0:
            self._name = name
        else:
            raise ValueError("Invalid name")


# Usage
person = Person("Jon")
print(person.name)  # Accessor: Output "Jon"
person.name = "Jonathan"  # Mutator
print(person.name)  # Output "Jon"

person.name = ""  # Raises ValueError
