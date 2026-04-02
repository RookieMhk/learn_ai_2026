import sys
import array
import numpy as np
import timeit

# 生成测试数据
data_source = list(range(1_000_000))

# 方案1：Python list
def test_list():
    return [float(x) for x in data_source]

# 方案2：tuple
def test_tuple():
    return tuple(float(x) for x in data_source)

# 方案3：array.array
def test_array():
    return array.array('f', [float(x) for x in data_source])

# 方案4：numpy.ndarray
def test_numpy():
    return np.arange(1_000_000, dtype=np.float32)

# 性能测试
for test_func in [test_list, test_array, test_numpy]:
    result = test_func()
    print(f"{test_func.__name__}:")
    print(f"  内存占用: {sys.getsizeof(result) / 1024**2:.2f} MB")
    print(f"  执行时间: {timeit.timeit(test_func, number=10):.4f} 秒\n")

import numpy as np

class RingBuffer:
    """
    高效环形缓冲区实现
    
    功能：
    - 固定大小，新数据覆盖旧数据
    - O(1) 时间复杂度的插入和读取
    - 支持批量获取最近N个数据点
    """
    
    def __init__(self, size, dtype=np.float32):
        self.size = size
        self.dtype = dtype
        self.buffer = np.zeros(size, dtype=dtype)
        self.pointer = 0
        self.filled = False
    
    def append(self, value):
        """添加单个数据点"""
        self.buffer[self.pointer] = value
        self.pointer = (self.pointer + 1) % self.size
        if self.pointer == 0:
            self.filled = True
    
    def extend(self, values):
        """批量添加数据"""
        values = np.asarray(values, dtype=self.dtype)
        n = len(values)
        
        if n >= self.size:
            # 如果数据量超过缓冲区大小，只保留最后size个
            self.buffer[:] = values[-self.size:]
            self.pointer = 0
            self.filled = True
            return
        
        remaining = self.size - self.pointer
        if remaining >= n:
            # 可以直接放入
            self.buffer[self.pointer:self.pointer + n] = values
            self.pointer += n
            if self.pointer == self.size:
                self.pointer = 0
                self.filled = True
        else:
            # 需要分割存放
            self.buffer[self.pointer:] = values[:remaining]
            self.buffer[:n - remaining] = values[remaining:]
            self.pointer = n - remaining
            self.filled = True
    
    def get_all(self):
        """获取所有有效数据"""
        if not self.filled:
            return self.buffer[:self.pointer]
        else:
            return np.concatenate([
                self.buffer[self.pointer:],
                self.buffer[:self.pointer]
            ])
    
    def get_recent(self, n):
        """获取最近的n个数据点"""
        if n >= self.size or not self.filled:
            return self.get_all()
        
        start = (self.pointer - n) % self.size
        if start < self.pointer:
            return self.buffer[start:self.pointer]
        else:
            return np.concatenate([
                self.buffer[start:],
                self.buffer[:self.pointer]
            ])
    
    def get_statistics(self):
        """获取统计信息"""
        data = self.get_all()
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'count': len(data)
        }

# 使用示例
buffer = RingBuffer(size=1000)

# 添加数据
for i in range(1500):
    buffer.append(i % 100)

# 获取统计信息
stats = buffer.get_statistics()
print(f"统计数据: {stats}")

# 获取最近100个点
recent = buffer.get_recent(100)
print(f"最近100个点的平均值: {np.mean(recent):.2f}")

print(buffer.get_all())
buffer.append(-1)
print(buffer.get_all())

from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        # 移动到最近使用位置
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)
        
        self.cache[key] = value
        
        # 超过容量删除最久未使用的
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

import heapq

class TopKSolution:
    def find_k_largest_heap(self, nums, k):
        """
        使用最小堆解决Top K问题
        时间复杂度: O(n log k)
        空间复杂度: O(k)
        """
        min_heap = []
        
        for num in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            else:
                if num > min_heap[0]:
                    heapq.heapreplace(min_heap, num)
        
        return sorted(min_heap, reverse=True)

import time
a = time.time()
li = TopKSolution().find_k_largest_heap(list(range(100_000_000)), 10)
print(time.time() - a)
print(li)

import re

line = "2023-12-25 15:30:00 [INFO] 1234567890 - 这是一条日志"
pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+) (\w+) (.+)'
match = re.match(pattern, line)
print(match.group(0))
print(match.groups())