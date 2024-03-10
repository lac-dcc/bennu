import sys

if __name__ == "__main__":

    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    else:
        raise Exception("Missing output file")
    
    f = open(out_file, "r")

    pr = False
    for l in f.readlines():
        l = l.strip()
        if l.split(" ")[0] == "avg" or pr == True:
            pr = True
            print(l)