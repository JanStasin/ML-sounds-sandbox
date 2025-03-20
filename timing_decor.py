import functools
import time

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
    return wrapper

# def progress_bar_decorator(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         total_items = None
#         if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
#             total_items = func.__annotations__['return'].__args__[0].__class__.__len__()
        
#         with tqdm(total=total_items, desc=f"{func.__name__} progress", unit='item', leave=True) as pbar:
#             result = func(*args, **kwargs)
            
#             # If the function yields items, update the progress bar
#             if hasattr(result, '__iter__'):
#                 for item in result:
#                     pbar.update(1)
#                     yield item
#             else:
#                 pbar.update(1)
        
#         return result
#     return wrapper