import numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
def swc2mat(image_ids, prediction, prediction_MAT, crop_size):
    prediction = prediction.astype(np.single)
    cdef float[:,:,:] prediction_view = prediction
    
    cdef int len_ids = len(image_ids)
    cdef int len_pred = np.shape(prediction)[1]
    
    cdef Py_ssize_t idx
    cdef Py_ssize_t index

    cdef int crop_size_view = crop_size
    
    cdef int max_x = crop_size_view - 1
    cdef int max_y = crop_size_view - 1
    cdef int max_z = crop_size_view - 1
    
    cdef int a = 0
    cdef int b = 0
    cdef int c = 0
    
    cdef double x = 0
    cdef double y = 0
    cdef double z = 0
    cdef double r = 0
    
    for idx in range(len_ids):
        for index in range(len_pred):
            if prediction_view[idx, index, 0] >= 0.5:
                x = (prediction_view[idx, index, 2]) * crop_size_view
                y = (prediction_view[idx, index, 3]) * crop_size_view
                z = (prediction_view[idx, index, 4]) * crop_size_view
                r = prediction_view[idx, index, 5] * crop_size_view

                for a in range(int(x-r-1), int(x+r+1)):
                    for b in range(int(y-r-1), int(y+r+1)):
                        for c in range(int(z-r-1), int(z+r+1)):
                            a = min(max_x, a)
                            b = min(max_y, b)
                            c = min(max_z, c)
                            a = max(0, a)
                            b = max(0, b)
                            c = max(0, c)

                            if ((x-a)*(x-a) + (y-b)*(y-b) + (z-c)*(z-c)) > r*r:
                                continue
                                        
                            prediction_MAT[idx, a, b, c] = 1
    return prediction_MAT