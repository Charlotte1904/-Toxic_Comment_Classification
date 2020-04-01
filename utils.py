import re
import json
import time
import functools
import logging

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        func_name = func.__name__
        if elapsed_time < 100: 
            print(f"Elapse time for {func_name}: {elapsed_time:0.2f} seconds")
        else:
            print(f"Elapse time for {func_name}: {elapsed_time/60:0.2f} minutes")
        return value
    return wrapper_timer

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, start_job_name="The Excution"):
        self._start_time = None
        self._toc_time = None
        self.start_job_name = start_job_name
        if start_job_name:
            print(f"Start {start_job_name} !")

    def start(self, start_job_name=""):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        if start_job_name:
            self.start_job_name = start_job_name
        self._start_time = time.perf_counter()
    
    def toc(self, toc_job_name):
        """Start a new timer"""
        if self._toc_time is None:
            elapsed_time = time.perf_counter() - self._start_time
        else:
            elapsed_time = time.perf_counter() - self._toc_time
            self._toc_time = time.perf_counter()
        if elapsed_time < 100: 
            print(f"{toc_job_name} takes {elapsed_time:0.2f} seconds")
        else:
            print(f"{toc_job_name} takes {elapsed_time/60:0.2f} minutes")

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        if elapsed_time < 100: 
            print(f"{self.start_job_name} takes {elapsed_time:0.2f} seconds")
        else:
            print(f"{self.start_job_name} takes {elapsed_time/60:0.2f} minutes")

        self._start_time = None
