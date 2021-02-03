import cv2
import numpy as np 

print("[CUDA TEST] Check CUDA enable device")
count = cv2.cuda.getCudaEnabledDeviceCount()
print("CUDA enable device count : %s \n" % count)

print("[CUDA TEST] test CUDA upload & download")
npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
cuMat = cv2.cuda_GpuMat()
cuMat.upload(npMat)
print( "Uploaded & Downloaded Matrix is close : %s \n" % np.allclose(cuMat.download(), npMat))

print("[CUDA TEST] test CUDA interoperability")
npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)

cuMat = cv2.cuda_GpuMat()
cuMat.upload(npMat)
print("Upload pointer:", cuMat.cudaPtr())

stream = cv2.cuda_Stream()
print("CUDA stream pointer:", stream.cudaPtr())