from time import time


def stopwatch(target_function):
    def wrapper(*args, **kwargs):
        start = time()
        target_function(*args, **kwargs)
        ellipse = time() - start
        print('[time:] {:.3f}[s]'.format(ellipse))
        return target_function

    return wrapper
