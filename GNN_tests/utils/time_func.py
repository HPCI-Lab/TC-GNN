import time

# Start recording time
def start_time():
    return time.time()

# Stop recording and print
def stop_time(START_TIME, MESSAGE=None):
    STOP_TIME = time.time()
    if not MESSAGE:
        MESSAGE = "default message"
    print(f"  ---  {MESSAGE}  ---  {(STOP_TIME - START_TIME)} seconds.")
