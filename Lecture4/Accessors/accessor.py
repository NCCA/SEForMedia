#!/usr/bin/env -S uv run --script


class Person:
    def __init__(self, name):
        self._name = name  # The underscore signifies a "protected" variable

    # Getter method (Accessor)
    def get_name(self):
        return self._name

    # Setter method (Mutator)
    def set_name(self, name):
        if isinstance(name, str) and len(name) > 0:
            self._name = name
        else:
            raise ValueError("Invalid name")


# Usage
person = Person("Jon")
print(person.get_name())  # Accessor: Output "Jon"
person.set_name("Jonathan")  # Mutator
print(person.get_name())  # Output "Jonathan"
person._name = 1235  # No error
print(person.get_name())  # Output 1235
