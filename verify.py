import sys, string, os, glob

def run(command):
    output = os.popen(command).read().split()
    return int(output[1])

programs = ["ref/floyd-ref {}", "ref/johnson-ref {}", "omp/floyd-omp -p 8 {}", "omp/johnson-omp -p 8 {}"]

def verify(filename):
    ref = run(programs[0].format(filename))
    for program in programs[2:]:
        command = program.format(filename)
        ans = run(command)
        if ref != ans:
            print("Failed: {} returned {} instead of {}", command, ans, ref)
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
