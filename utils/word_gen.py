import sys
import random


if "__main__":
    if (len(sys.argv) != 3):
        print ("No input file? or number count really")
    try:
        int (sys.argv[2])
    except:
        print ("Not a number")
        sys.exit(1)

    with open ("output.txt", "w+") as output:
        with open(sys.argv[1], "r+") as f:
            lines = f.readlines()
            output.writelines(random.sample(lines, int(sys.argv[2])))
            

