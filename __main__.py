import subprocess
import keyboard
def screenshot():
    subprocess.run(["python3", "screenshot.py", ""])
if __name__ == '__main__':
    keyboard.add_hotkey("f1",screenshot)
    keyboard.wait()
    