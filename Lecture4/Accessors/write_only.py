#!/usr/bin/env -S uv run --script


class Secret:
    def __init__(self):
        self._secret_value = None

    @property
    def secret(self):
        raise AttributeError("This is write-only")

    @secret.setter
    def secret(self, value):
        self._secret_value = value


# Usage
s = Secret()
s.secret = "My secret"  # This works
print(s.secret)  # Raises AttributeError since no getter is defined
