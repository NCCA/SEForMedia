#!/usr/bin/env python
import random

# only do simple operations no division for now
ops = ["+", "-", "*"]


def get_question() -> str:
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    op = random.choice(ops)
    return f"{a} {op} {b}"


def get_number() -> int:
    while True:
        try:
            return int(input("Enter result: "))
        except ValueError:
            print("Please enter a valid number")


num_correct = 0

again = True
while again:
    for i in range(5):
        print(f"Question {i} of 5")
        question = get_question()
        print("What is the result of this expression?")
        print(f"{question}")
        result = eval(question)
        input_result = get_number()
        if result == input_result:
            print("Correct")
            num_correct += 1
        else:
            print("Incorrect")
            print(f"The correct answer is {result}")

    print(f"You got {num_correct} correct out of 5")
    while True:
        play_again = input("Do you want to play again? (yes/no) ")
        if play_again.lower() in ["yes", "no"]:
            break
    if play_again.lower() == "no":
        again = False
        print("Goodbye")
    else:
        num_correct = 0
        print("Starting again")
