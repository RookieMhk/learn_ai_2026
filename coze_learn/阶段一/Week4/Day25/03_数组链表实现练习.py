#!/usr/bin/env python3
"""
Day 25 代码实践：数组与链表操作
包含完整实现 + 性能测试
"""

import time
import random
from typing import List, Optional, TypeVar
from collections import defaultdict

# ========================
# Part 1: 数组操作
# ========================

class ArrayOperations:
    """数组操作封装"""
    
    @staticmethod
    def create(size: int, fill_value=0):
        """创建数组"""
        return [fill_value] * size
    
    @staticmethod
    def access(arr: List, index: int):
        """O(1) 随机访问"""
        return arr[index]
    
    @staticmethod
    def insert(arr: List, index: int, value) -> List:
        """O(n) 插入"""
        arr.insert(index, value)
        return arr
    
    @staticmethod
    def delete(arr: List, index: int):
        """O(n) 删除"""
        arr.pop(index)
        return arr
    
    @staticmethod
    def find(arr: List, value) -> int:
        """O(n) 查找"""
        try:
            return arr.index(value)
        except ValueError:
            return -1
    
    @staticmethod
    def traverse(arr: List):
        """遍历"""
        return [x for x in arr]


class DynamicArray:
    """
    动态数组实现（类似Python list）
    复杂度分析：
    - 访问: O(1)
    - 追加: O(1) amortized
    - 插入: O(n)
    - 删除: O(n)
    """
    
    def __init__(self, initial_capacity=4):
        self.capacity = initial_capacity
        self.size = 0
        self.data = [None] * self.capacity
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        if 0 <= index < self.size:
            return self.data[index]
        raise IndexError("Index out of range")
    
    def append(self, item):
        """O(1) amortized 追加"""
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        self.data[self.size] = item
        self.size += 1
    
    def _resize(self, new_capacity):
        """扩容"""
        new_data = [None] * new_capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
        self.capacity = new_capacity
    
    def insert(self, index, item):
        """O(n) 插入"""
        if index < 0 or index > self.size:
            raise IndexError("Index out of range")
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]
        self.data[index] = item
        self.size += 1
    
    def delete(self, index):
        """O(n) 删除"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        item = self.data[index]
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        self.size -= 1
        return item


# ========================
# Part 2: 链表实现
# ========================

class ListNode:
    """链表节点"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"


class LinkedList:
    """
    单链表实现
    复杂度分析：
    - 访问: O(n)
    - 头部插入: O(1)
    - 尾部插入: O(n)
    - 插入/删除(已知位置): O(1)
    - 查找: O(n)
    """
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def is_empty(self):
        return self.head is None
    
    def append(self, val):
        """尾部追加 O(n)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = new_node
        self.size += 1
    
    def prepend(self, val):
        """头部插入 O(1)"""
        new_node = ListNode(val, self.head)
        self.head = new_node
        self.size += 1
    
    def insert_after(self, prev_node: ListNode, val):
        """在指定节点后插入 O(1)"""
        if not prev_node:
            raise ValueError("Previous node cannot be None")
        new_node = ListNode(val, prev_node.next)
        prev_node.next = new_node
        self.size += 1
    
    def delete_node(self, node: ListNode):
        """删除指定节点 O(1) - 注意：需要能访问到前一节点"""
        if not node or not node.next:
            return False
        node.val = node.next.val
        node.next = node.next.next
        self.size -= 1
        return True
    
    def delete_value(self, val):
        """删除第一个匹配的值 O(n)"""
        if not self.head:
            return False
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        curr = self.head
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                self.size -= 1
                return True
            curr = curr.next
        return False
    
    def find(self, val) -> Optional[ListNode]:
        """查找节点 O(n)"""
        curr = self.head
        while curr:
            if curr.val == val:
                return curr
            curr = curr.next
        return None
    
    def traverse(self) -> List:
        """遍历所有节点"""
        result = []
        curr = self.head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result
    
    def reverse(self):
        """反转链表 O(n)"""
        prev = None
        curr = self.head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        self.head = prev
    
    def has_cycle(self) -> bool:
        """检测环 O(n)"""
        if not self.head:
            return False
        slow = self.head
        fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False


class DoublyLinkedList:
    """双向链表"""
    
    class Node:
        def __init__(self, val):
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, val):
        """尾部追加 O(1)"""
        new_node = self.Node(val)
        if not self.tail:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def prepend(self, val):
        """头部插入 O(1)"""
        new_node = self.Node(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
    
    def delete(self, node):
        """删除节点 O(1)"""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        self.size -= 1


# ========================
# Part 3: LRU缓存实现
# ========================

class LRUCache:
    """
    LRU缓存：哈希表 + 双向链表
    时间复杂度：O(1) 所有操作
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        # 双向链表: head <-> ... <-> tail
        self链表 = DoublyLinkedList()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 移到链表头部（最近使用）
        self链表.delete(node)
        self链表.prepend(node.val)
        return node.val[1]  # 返回 value
    
    def put(self, key: int, value: int):
        if key in self.cache:
            # 更新并移到头部
            self链表.delete(self.cache[key])
        else:
            if len(self.cache) >= self.capacity:
                # 删除最久未使用的（链表尾部）
                lru = self链表.tail.val
                self链表.delete(self链表.tail)
                del self.cache[lru[0]]
        
        self链表.prepend((key, value))
        self.cache[key] = self链表.head


# ========================
# Part 4: 性能测试
# ========================

def benchmark_array_operations(size=10000, iterations=1000):
    """测试数组操作的性能"""
    print(f"\n{'='*50}")
    print(f"数组性能测试 (size={size}, iterations={iterations})")
    print(f"{'='*50}")
    
    arr = ArrayOperations.create(size, 0)
    
    # 随机访问测试
    indices = [random.randint(0, size-1) for _ in range(iterations)]
    start = time.time()
    for idx in indices:
        _ = arr[idx]
    print(f"随机访问 {iterations} 次: {(time.time()-start)*1000:.2f}ms")
    
    # 插入测试（小规模）
    small_arr = list(range(100))
    start = time.time()
    for _ in range(100):
        small_arr.insert(50, 0)
    print(f"插入 100 次: {(time.time()-start)*1000:.2f}ms")
    
    # 动态数组追加测试
    dyn_arr = DynamicArray()
    start = time.time()
    for i in range(size):
        dyn_arr.append(i)
    print(f"动态数组追加 {size} 次: {(time.time()-start)*1000:.2f}ms")


def benchmark_linkedlist_operations(size=10000, iterations=1000):
    """测试链表操作的性能"""
    print(f"\n{'='*50}")
    print(f"链表性能测试 (size={size}, iterations={iterations})")
    print(f"{'='*50}")
    
    ll = LinkedList()
    for i in range(size):
        ll.append(i)
    
    # 头部插入测试
    start = time.time()
    for _ in range(min(1000, size//10)):
        ll.prepend(0)
    print(f"头部插入 1000 次: {(time.time()-start)*1000:.2f}ms")
    
    # 查找测试
    search_vals = [random.randint(0, size-1) for _ in range(iterations)]
    start = time.time()
    for val in search_vals:
        _ = ll.find(val)
    print(f"查找 {iterations} 次: {(time.time()-start)*1000:.2f}ms")


def compare_array_vs_linkedlist():
    """数组 vs 链表对比"""
    print(f"\n{'='*50}")
    print("数组 vs 链表 对比测试")
    print(f"{'='*50}")
    
    size = 1000
    insert_pos = size // 2
    
    # 数组插入
    arr = list(range(size))
    start = time.time()
    for _ in range(100):
        arr.insert(insert_pos, 0)
    arr_time = (time.time() - start) * 1000
    
    # 链表插入（需先找到位置）
    ll = LinkedList()
    for i in range(size):
        ll.append(i)
    start = time.time()
    for _ in range(100):
        curr = ll.head
        for _ in range(insert_pos):
            curr = curr.next
        ll.insert_after(curr, 0)
    ll_time = (time.time() - start) * 1000
    
    print(f"数组插入 100 次: {arr_time:.2f}ms")
    print(f"链表插入 100 次: {ll_time:.2f}ms")
    print(f"结论: 链表插入需先遍历查找，整体更慢")


# ========================
# Part 5: AI场景应用
# ========================

class AIDataStructures:
    """AI场景数据结构应用"""
    
    @staticmethod
    def image_as_array():
        """
        图像存储为数组
        模拟: 224x224 RGB图像
        """
        import numpy as np
        
        # 模拟图像: [H, W, C]
        h, w, c = 224, 224, 3
        image = np.random.rand(h, w, c)
        
        # 随机访问像素
        y, x = 100, 150
        pixel = image[y, x]  # O(1) 访问
        print(f"图像尺寸: {image.shape}")
        print(f"像素 [{y}, {x}] = {pixel}")
        
        # 切片提取区域
        region = image[50:100, 100:150]  # O(h*w) 连续内存访问
        print(f"区域形状: {region.shape}")
        
        return image
    
    @staticmethod
    def rnn_hidden_state():
        """
        RNN隐藏状态传递（链表式）
        """
        class RNNCell:
            def __init__(self, hidden_dim):
                self.hidden_dim = hidden_dim
                self.h_prev = None
            
            def forward(self, x, h_prev=None):
                """前向传播"""
                self.h_prev = h_prev
                # 简化的RNN计算
                h = np.tanh(np.random.rand(self.hidden_dim))
                return h
        
        hidden_dim = 256
        time_steps = 10
        
        rnn = RNNCell(hidden_dim)
        h = None
        
        print(f"\nRNN时间步隐藏状态传递:")
        for t in range(time_steps):
            x_t = np.random.rand(hidden_dim)
            h = rnn.forward(x_t, h)
            print(f"  t={t}: hidden_state shape = {h.shape}")
        
        return h
    
    @staticmethod
    def word_embedding_lookup():
        """
        词嵌入查找（数组 + 哈希表）
        """
        import numpy as np
        
        vocab = {
            "hello": 0,
            "world": 1,
            "ai": 2,
            "data": 3,
            "structure": 4
        }
        
        vocab_size = len(vocab)
        embedding_dim = 128
        
        # 嵌入矩阵: [vocab_size, embedding_dim]
        embeddings = np.random.rand(vocab_size, embedding_dim)
        
        word = "hello"
        word_id = vocab[word]
        embedding = embeddings[word_id]  # O(1) 查找
        
        print(f"\n词嵌入查找:")
        print(f"  词: '{word}' -> ID: {word_id}")
        print(f"  嵌入向量形状: {embedding.shape}")
        
        return embeddings, vocab


# ========================
# 主函数
# ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 25: 数组与链表 代码实践")
    print("=" * 60)
    
    # 性能测试
    benchmark_array_operations()
    benchmark_linkedlist_operations()
    compare_array_vs_linkedlist()
    
    # AI场景应用
    print("\n" + "=" * 60)
    print("AI场景应用演示")
    print("=" * 60)
    
    AIDataStructures.image_as_array()
    AIDataStructures.rnn_hidden_state()
    AIDataStructures.word_embedding_lookup()
    
    # LRU缓存测试
    print("\n" + "=" * 60)
    print("LRU缓存测试")
    print("=" * 60)
    
    cache = LRUCache(3)
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    print(f"put(1,10), put(2,20), put(3,30)")
    print(f"get(1) = {cache.get(1)}")  # 返回10
    cache.put(4, 40)  # 淘汰key=2
    print(f"put(4,40) -> 淘汰最久未使用的")
    print(f"get(2) = {cache.get(2)}")  # 返回-1（已淘汰）
    print(f"get(3) = {cache.get(3)}")  # 返回30
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
