import torch
import time

# Large matrix for stress test
size = 9000  

# CPU test
x_cpu = torch.randn(size, size)
start = time.time()
y_cpu = x_cpu @ x_cpu
end = time.time()
print("CPU time:", end - start, "seconds")

# GPU test (if available)
if torch.cuda.is_available():
    x_gpu = torch.randn(size, size, device="cuda")
    torch.cuda.synchronize()  # clear previous tasks
    start = time.time()
    y_gpu = x_gpu @ x_gpu
    torch.cuda.synchronize()  # wait for GPU to finish
    end = time.time()
    print("GPU time:", end - start, "seconds")
    print("Allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
    print("Reserved:", torch.cuda.memory_reserved()/1024**2, "MB")

