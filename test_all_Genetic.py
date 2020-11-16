import subprocess
import re
numSuccess = 0
numFailures = 0
for i in range(100):
    process = subprocess.run(
        ["python3", "game.py", "--l", str(i), "--agent", "Genetic"], stdout=subprocess.PIPE)
    output = process.stdout.decode("utf-8")
    # print(output)
    if "GAME WON IN" in output:
        numSuccess += 1
    elif "SOLUTION NOT FOUND" in output:
        numFailures += 1

print("Successes: ", numSuccess)
print("Failures: ", numFailures)