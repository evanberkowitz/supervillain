import time

class Timer:
    
    r'''
    A simple timer which provides a context manager for use with ``with`` statements.

    Parameters
    ----------
        log: A logger's convenience function, such as ``.debug`` or ``.info``.
            Where to log the timer.
        message: str
            The message on entry and exit from the timer's context, unless stop_message is also provided, in which case this is only on entry.
        stop_message: str
            An optionally different message to print on completion.
        time: a function which returns times that can be subtracted to get an interval.
            Examples include ``time.perf_counter`` or ``time.process_time`` as `in the standard library <https://docs.python.org/3/library/time.html#time.perf_counter>`_.
        per: int
            How many to divide by to determine the marginal cost.

    For example,
    
    >>> with Timer(logger.debug, 'Task', per=10):
    >>>     time.sleep(0.3, time=time.process_time)
    2023-04-27 22:39:18,076 root      DEBUG Task ...
    2023-04-27 22:39:18,383 root      DEBUG ... Task [0.001954 seconds] (0.000195 seconds each)
    >>> with Timer(logger.debug, 'Begin', 'End', time=time.process_time):
    >>>     time.sleep(0.3)
    2023-04-27 22:39:51,286 root      DEBUG Begin ...
    2023-04-27 22:39:51,604 root      DEBUG ... End [0.001070 seconds]
    >>> with supervillain.performance.Timer(logger.debug, 'With perf_counter', time=time.perf_counter):
    >>>     time.sleep(0.3)
    2023-04-27 22:45:55,955 root      DEBUG With perf_counter ...
    2023-04-27 22:45:56,258 root      DEBUG ... With perf_counter [0.300386 seconds]

    '''
    
    def __init__(self, 
                 log, 
                 message, 
                 stop_message=None, 
                 time=time.perf_counter,
                 per=1
                ):

        self.time = time
        self.fmt  = '8f'
        self.log = log

        self.per = per
        
        if stop_message:
            self.start_msg = message
            self.stop_msg  = stop_message
        else:
            self.start_msg = message
            self.stop_msg  = message

        self.strt_time = None
        self.stop_time = None
    
    def start(self):
        r'''Start the timer.'''
        self.strt_time = self.time()
        
    def stop(self):
        r'''Stop the timer.'''
        self.stop_time = self.time()

    def elapsed(self):
        r'''
        The difference between the stop and start time.  If the timer was never :func:`~.Timer.start`\ ed, 0; if the timer was never :func:`~.Timer.stop`\ ped, stop now and and subtract.
        '''
        if self.strt_time is None:
            return 0.
        if self.stop_time is None:
            self.stop()
        return self.stop_time - self.strt_time

    
    def __enter__(self):
        self.log(f'{self.start_msg} ...')
        self.start()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        diff = self.elapsed()
        self.log(f'... {self.stop_msg} [{diff:{self.fmt}} seconds]' + (f' ({diff/self.per:{self.fmt}} seconds each * {self.per})' if self.per > 1 else ''))
