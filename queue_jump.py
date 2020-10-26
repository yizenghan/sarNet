import pynvml
import time
def main():
    check_gpu_memory()

def check_gpu_memory():
    pynvml.nvmlInit()

    gpulist = [0,3]
    deviceCount = pynvml.nvmlDeviceGetCount()
    
    print('There are {} visiable GPUs'.format(deviceCount))
    while 1>0:
        
        num_free = 0
        for i in range(len(gpulist)):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpulist[i])
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_free = meminfo.free/1024**2
                print('Free memory on GPU device{} is {}M'.format(i, mem_free))
                if mem_free > 8e3:
                    num_free += 1
        if num_free == len(gpulist):
            break
        else:
            print('Not enough memory on GPU, wait for 10s and retry!')
            time.sleep(10)

if __name__ == '__main__':
    main()