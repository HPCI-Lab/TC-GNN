import time

# Start recording time
def start_time():
    return time.time()

# Stop recording and print
def stop_time(START_TIME, MESSAGE):
    print(f"--- {MESSAGE} --- {(time.time() - START_TIME)} seconds.")
