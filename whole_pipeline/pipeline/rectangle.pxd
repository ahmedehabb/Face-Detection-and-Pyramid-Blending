cdef class Rectangle:
    cdef public int x, y, rect_width, rect_height, rect_weight
    cpdef public Rectangle scale_rectangle(self, float scale_factor)
