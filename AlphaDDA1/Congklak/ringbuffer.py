#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np

class RingBuffer:
    def __init__(self, buf_size):
        self.size = buf_size
        self.buf = []
        for _ in range(self.size):
            self.buf.append([])
        self.start = 0
        self.end = 0

    def add(self, el):
        self.buf[self.end] = el
        self.end = (self.end + 1) % self.size
        if self.end == self.start:
            self.start = (self.start + 1) % self.size

    def Get_buffer(self):
        array = []
        for i in range(self.size):
            buf_num = (self.end - i) % self.size
            array.append(self.buf[buf_num])
        return array

    def Get_buffer_start_end(self):
        array = []
        for i in range(self.size):
            buf_num = (self.start + i) % self.size
            # Fix: check if it's an empty list to avoid numpy broadcasting errors
            # Explicitly check length to avoid numpy elementwise comparison issues
            if isinstance(self.buf[buf_num], list) and len(self.buf[buf_num]) == 0:
                # Stop gathering if we hit an empty initialized slot
                return array
            array.append(self.buf[buf_num])
        return array

    def get(self):
        val = self.buf[self.start]
        self.start =(self.start + 1) % self.size
        return val
