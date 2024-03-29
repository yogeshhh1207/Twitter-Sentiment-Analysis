import os
import subprocess

subprocess.call(["find", "database", "-type", "f", "-name", "*.csv", "-delete"])

subprocess.call(["find", "results", "-type", "f", "-name", "*.csv", "-delete"])