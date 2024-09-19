#!/usr/bin/env python
import sys


def main():
    for arg in sys.argv[1:]:  # sys.argv[0] is the name of the script
        print(f"arg: {arg}")
        if "-h" in arg or "--help" in arg:
            print("found helps")
        elif "-o" in arg or "--output" in arg:
            print("found output")
        elif "-r" in arg or "--rows" in arg:
            print("found rows")
        elif "-c" in arg or "--columns" in arg:
            print("found columns")
        elif "-s" in arg or "--separator" in arg:
            print("found separator")
        else:
            print(f" unknown argument {arg}")


if __name__ == "__main__":
    main()
