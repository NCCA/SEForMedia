#!/usr/bin/env python

import click


@click.command()
@click.option("-o", "--output", help="The name of the output file", required=True)
def main(output, rows, columns, separator):
    print(f"output: {output}")


if __name__ == "__main__":
    main()
