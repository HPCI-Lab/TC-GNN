import time

# Start recording time
def start_time():
    return time.time()

# Stop recording and print
def stop_time(START_TIME, MESSAGE):
    STOP_TIME = time.time()
    print(f"  ---  {MESSAGE}  ---  {(STOP_TIME - START_TIME)} seconds.")
