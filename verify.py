import sys, string, os, glob


def verify(filename):
    # TODO: change program names
    outputRef = os.popen("ref/floyd-ref " + filename).read().split("\n")
    output = os.popen("ref/johnson-ref " + filename).read().split("\n")
    for (i, (lineRef, line)) in enumerate(zip(outputRef, output)):
        if i < 4:
            continue
        distsRef = lineRef.rstrip().split(" ")
        dists = line.rstrip().split(" ")
        for (j, (distRef, dist)) in enumerate(zip(distsRef, dists)):
            if distRef != dist:
                print("Mismatch in {} on distance from {} to {}: ref got {}, instead got {}".format(filename, i, j, distRef, dist))
                return False
    print("Passed: {}".format(filename))
    return True


def main():

    print("")
    print("Testing...")
    print("-----------------------------------------------")
    numFailed = 0

    for filename in glob.glob("input/*.txt"):
        passed = verify(filename)
        if not passed:
            numFailed += 1
    print("-----------------------------------------------")
    if(numFailed == 0):
        print("All tests passed!")
    else:
        print("%d test(s) failed." % (numFailed))
    print("")

if __name__ == '__main__':
    main()
