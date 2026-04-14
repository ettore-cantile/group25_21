import subprocess
import sys

# List of the scripts in the original order
original_scripts = [
    "compute_fp.py",
    "compute_kmeans_2d.py",
    "compute_kmeans.py",
    "compute_projection.py",
    "compute_relationship.py"
]


# Reverse the order of the list
reversed_scripts = original_scripts[::-1]

print("Starting the execution of the scripts in reverse order...\n")
print("-" * 40)

for script in reversed_scripts:
    print(f"Executing: {script}")
    try:
        # sys.executable ensures the current Python interpreter is used
        subprocess.run([sys.executable, script], check=True)
        print(f"-> {script} completed successfully.\n")
        
    except subprocess.CalledProcessError as e:
        # Catches errors if the executed script crashes or returns a non-zero exit code
        print(f"\n[ERROR] The execution of {script} failed (Exit code: {e.returncode}).")
        print("Stopping the process.")
        break # Stop the loop so subsequent scripts don't run
        
    except FileNotFoundError:
        # Catches the error if the script file doesn't exist in the directory
        print(f"\n[ERROR] Could not find the file '{script}'. Make sure it is in the same directory.")
        print("Stopping the process.")
        break

print("-" * 40)
print("Process finished.")
