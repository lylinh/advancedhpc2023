import numba
from numba import cuda

# Detect device
numba.cuda.detect()

# Found only 1 device, so choose the device_id = 0 
device_id = 0 
device = numba.cuda.select_device( device_id )

# Get the GPU device information
device_name = device.name.decode('utf-8')
device_multiprocessor_count = device.MULTIPROCESSOR_COUNT
device_memory_size = cuda.current_context().get_memory_info().total
   
# Display the device information
print(f"Device Name: {device_name}")
print(f"Device Multiprocessor Count: {device_multiprocessor_count}")
print(f"Device Memory Size: {device_memory_size} ")