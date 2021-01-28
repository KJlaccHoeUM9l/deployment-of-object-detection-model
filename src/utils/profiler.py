import time


class LogDuration:
    def __init__(self, name='', logger=None, print_log=True):
        self.name = name
        self.logger = logger
        self.print_log = print_log

    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self._startTime
        fps = 1. / elapsed_time
        if self.logger is not None:
            self.logger[self.name] = fps
        if self.print_log:
            print("[{}] fps: {:.3f}".format(self.name, fps))
