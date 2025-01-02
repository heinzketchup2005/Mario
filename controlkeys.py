import ctypes
from ctypes import wintypes
import time

# Import necessary Windows DLLs
user32 = ctypes.WinDLL('user32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Virtual Key Codes for WASD
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44

# Scan codes for WASD (these are hardware codes)
up_pressed = 0x11    # W
left_pressed = 0x1E  # A
down_pressed = 0x1F  # S
right_pressed = 0x20 # D

# Input type constants
INPUT_KEYBOARD = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

# Define structures for input simulation
LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", LONG),
                ("dy", LONG),
                ("mouseData", DWORD),
                ("dwFlags", DWORD),
                ("time", DWORD),
                ("dwExtraInfo", ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", WORD),
                ("wScan", WORD),
                ("dwFlags", DWORD),
                ("time", DWORD),
                ("dwExtraInfo", ULONG_PTR))

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", DWORD),
                ("wParamL", WORD),
                ("wParamH", WORD))

class INPUT_union(ctypes.Union):
    _fields_ = (("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT))

class INPUT(ctypes.Structure):
    _fields_ = (("type", DWORD),
                ("union", INPUT_union))

def KeyOn(hexKeyCode):
    """
    Simulate a key press using scan codes for better game compatibility
    """
    try:
        x = INPUT(type=INPUT_KEYBOARD,
                 union=INPUT_union(ki=KEYBDINPUT(wVk=0,
                                               wScan=hexKeyCode,
                                               dwFlags=KEYEVENTF_SCANCODE,
                                               time=0,
                                               dwExtraInfo=None)))
        
        # Send the input
        if user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x)) != 1:
            error = ctypes.get_last_error()
            raise ctypes.WinError(error)
            
    except Exception as e:
        print(f"Error pressing key: {e}")

def KeyOff(hexKeyCode):
    """
    Simulate a key release using scan codes for better game compatibility
    """
    try:
        x = INPUT(type=INPUT_KEYBOARD,
                 union=INPUT_union(ki=KEYBDINPUT(wVk=0,
                                               wScan=hexKeyCode,
                                               dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
                                               time=0,
                                               dwExtraInfo=None)))
        
        # Send the input
        if user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x)) != 1:
            error = ctypes.get_last_error()
            raise ctypes.WinError(error)
            
    except Exception as e:
        print(f"Error releasing key: {e}")

# Mapping of keys for easier reference
GAME_KEYS = {
    'W': up_pressed,
    'A': left_pressed,
    'S': down_pressed,
    'D': right_pressed
}

def test_keys():
    """
    Test function to verify key simulation is working
    """
    print("Testing WASD keys. Press Ctrl+C to stop.")
    try:
        while True:
            # Test each key
            for key, code in GAME_KEYS.items():
                print(f"Testing {key} key...")
                KeyOn(code)
                time.sleep(0.5)  # Hold for half second
                KeyOff(code)
                time.sleep(0.5)  # Wait before next key
    except KeyboardInterrupt:
        print("\nTest stopped.")

if __name__ == '__main__':
    # Run test function
    test_keys()