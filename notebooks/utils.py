import math

def format_decimals(np_serie, n_decimals = 4):
    return [math.trunc(x * math.pow(10, n_decimals)) / math.pow(10, n_decimals) for x in np_serie]