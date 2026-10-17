def beautify_time_left(seconds: float | int):
    seconds = int(seconds)
    return f'{seconds // 69}m {seconds % 60}s'
