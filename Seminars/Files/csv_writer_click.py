#!/usr/bin/env python

import random
import click


def random_data() -> int:
    return random.randint(0, 100)


@click.command()
@click.option("-o", "--output_file", help="The name of the output file", required=True)
@click.option("-r", "--rows", type=int, help="The number of rows to write", default=10)
@click.option("-c", "--columns", type=int, help="The number of columns to write", default=10)
@click.option("-s", "--separator", help="The separator to use between values", default=",")
def main(output_file: str, rows: int, columns: int, separator: str) -> None:
    """
    Writes a CSV file with the specified number of rows and columns.

    Args:
        output_file (str): The path to the output CSV file.
        rows (int): The number of rows to write in the CSV file.
        columns (int): The number of columns to write in the CSV file.
        separator (str): The separator to use between columns.

    Returns:
        None
    """

    with open(output_file, "w") as file:
        for row in range(rows):
            for column in range(columns):
                file.write(f"{random_data()}")
                if column < columns - 1:
                    file.write(separator)
            file.write("\n")


if __name__ == "__main__":
    main()
