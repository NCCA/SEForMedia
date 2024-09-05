#!/usr/bin/env python

format = "png"

match format:
    case "png":
        print("PNG format selected")
    case "jpeg":
        print("JPEG format selected")
    case "gif":
        print("GIF format selected")
    case _:
        print("Unknown format selected")
