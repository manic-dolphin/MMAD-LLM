import time

def time_counter(begin_time, end_time):
    
    run_time = round(end_time - begin_time)
    
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    
    return hour, minute, second