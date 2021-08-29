cimport cython
from cython.parallel import prange, parallel, threadid
from libc.string cimport memcpy
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef int _apply_select_features(int th_id, int num_thread, const float[:,:,:,:] input, const int[:, :, :, :] offset, float[:,:,:,:] output) nogil:
    cdef int n, c, h, w
    cdef int oh, ow, _i, _j, _c
    cdef int _x, _y
    cdef int NTH, NID
    NTH = num_thread
    NID = th_id
    n = input.shape[0]
    c = input.shape[1]
    h = input.shape[2]
    w = input.shape[3]

    oh = offset.shape[1]
    ow = offset.shape[2]
    for _c in range(c):
        if _c % NTH != NID:
            continue

        for _i in range(oh):
            for _j in range(ow):
                _x = offset[0, _i, _j, 1]
                _y = offset[0, _i, _j, 0]
                if(_x >= 0 and _x < w and _y >= 0 and _y < h):
                        output[0, _c, _i, _j] = input[0, _c, _y, _x]

cpdef apply_select_features(const float [:,:,:,:] input, const int[:, :, :, :] offset, const int thread, float[:,:,:,:] output):
    cdef int th
    cdef int _thread
    _thread = thread
    for th in prange(_thread, num_threads=_thread, nogil=True):
        _apply_select_features(th, _thread, input, offset, output)
