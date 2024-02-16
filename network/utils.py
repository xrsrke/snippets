import math

def calculate_dimensions(sizes):
    MB_TO_BYTES = 2**20
    GB_TO_BYTES = 2**30
    
    results = []
    
    for size, unit in sizes:
        if unit == "MB":
            size_in_bytes = size * MB_TO_BYTES
        elif unit == "GB":
            size_in_bytes = size * GB_TO_BYTES
        else:
            raise ValueError("Unknown unit: " + unit)
        
        # Calculate N (assuming M = N for a square tensor)
        N = int(math.sqrt(size_in_bytes / 4))
        M = N  # Since we are assuming a square tensor
        results.append((M, N))
    
    return results

def bytes_to_nice_format(size_in_bytes):
    GB_TO_BYTES = 2**30
    MB_TO_BYTES = 2**20
    
    if size_in_bytes < GB_TO_BYTES:
        size_in_mb = size_in_bytes / MB_TO_BYTES
        return f"{size_in_mb:.2f} MB"
    else:
        size_in_gb = size_in_bytes / GB_TO_BYTES
        return f"{size_in_gb:.2f} GB"