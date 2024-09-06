#!/usr/bin/env python


def get_int():
    while True:
        try:
            userInput = int(input("please enter an int >"))
        except ValueError:
            print("Not an integer! Try again.")
            continue
        else:
            return userInput
            break


end = get_int()
for i in range(0, end):
    print(i)
