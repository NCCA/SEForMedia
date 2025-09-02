#!/usr/bin/env -S uv run --script


class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name  # No setter defined


# Usage
person = Person("Jon")
print(person.name)  # Output: Jon

person.name = (
    "Jonathan"  # This would raise an AttributeError since the setter is not defined
)
