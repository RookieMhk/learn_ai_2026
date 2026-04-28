# Day 25 LeetCode题解：数组与链表

---

## 题目1：两数之和（LeetCode #1）

**难度**：简单  
**链接**：https://leetcode.com/problems/two-sum/

### 题目描述
给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出和为目标值 `target` 的那两个整数，并返回它们的数组下标。

### 示例
```
输入: nums = [2,7,11,15], target = 9
输出: [0,1]
解释: 因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

输入: nums = [3,2,4], target = 6
输出: [1,2]
```

### 解题思路

**核心思想**：空间换时间

- 使用哈希表（数组实现的哈希表）存储已遍历的元素及其索引
- 遍历时检查 `target - current_element` 是否在哈希表中
- 如果在，直接返回两个索引
- 如果不在，将当前元素加入哈希表

### 代码实现

```python
def two_sum(nums: List[int], target: int) -> List[int]:
    """
    时间复杂度: O(n) - 只需遍历一次数组
    空间复杂度: O(n) - 哈希表存储
    """
    hash_map = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    
    return []  # 无解（题目保证有解）


# 测试
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),
    ]
    
    for nums, target, expected in test_cases:
        result = two_sum(nums, target)
        print(f"nums={nums}, target={target}")
        print(f"  预期: {expected}, 结果: {result}, 通过: {result == expected}\n")
```

### 复杂度分析

| 指标 | 复杂度 | 说明 |
|-----|-------|-----|
| 时间 | O(n) | 单次遍历 |
| 空间 | O(n) | 哈希表 |

### 为什么用哈希表？
- 数组查找：O(n)
- 哈希表查找：O(1) 平均
- 将时间从 O(n²) 优化到 O(n)

---

## 题目2：合并两个有序链表（LeetCode #21）

**难度**：简单  
**链接**：https://leetcode.com/problems/merge-two-sorted-lists/

### 题目描述
将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

### 示例
```
输入: l1 = [1,2,4], l2 = [1,3,4]
输出: [1,1,2,3,4,4]

输入: l1 = [], l2 = []
输出: []
```

### 解题思路

**核心技巧**：Dummy节点（哑节点）

- 创建一个虚拟头节点 `dummy`，简化边界处理
- 使用双指针同时遍历两个链表
- 每次选择较小的节点加入结果链表
- 最终处理剩余部分

### 代码实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    时间复杂度: O(n + m) - n和m是两个链表的长度
    空间复杂度: O(1) - 只用指针，不创建新节点
    """
    # 创建哑节点，简化头节点处理
    dummy = ListNode(-1)
    curr = dummy
    
    # 双指针遍历
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    # 连接剩余部分
    curr.next = l1 if l1 else l2
    
    return dummy.next


def list_to_linkedlist(values: list) -> Optional[ListNode]:
    """辅助函数：将列表转换为链表"""
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for val in values[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head


def linkedlist_to_list(head: Optional[ListNode]) -> list:
    """辅助函数：将链表转换为列表"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# 测试
if __name__ == "__main__":
    test_cases = [
        ([1, 2, 4], [1, 3, 4], [1, 1, 2, 3, 4, 4]),
        ([], [], []),
        ([], [0], [0]),
    ]
    
    print("合并两个有序链表测试:")
    for vals1, vals2, expected in test_cases:
        l1 = list_to_linkedlist(vals1)
        l2 = list_to_linkedlist(vals2)
        result = merge_two_lists(l1, l2)
        result_list = linkedlist_to_list(result)
        print(f"  {vals1} + {vals2}")
        print(f"  预期: {expected}, 结果: {result_list}, 通过: {result_list == expected}\n")
```

### 复杂度分析

| 指标 | 复杂度 | 说明 |
|-----|-------|-----|
| 时间 | O(n + m) | 遍历两个链表各一次 |
| 空间 | O(1) | 只用指针，没有创建新节点 |

### Dummy节点技巧详解

```python
# 不使用dummy（需要特殊处理头节点）
if l1.val <= l2.val:
    result_head = l1
    l1 = l1.next
else:
    result_head = l2
    l2 = l2.next

# 使用dummy（统一处理）
dummy = ListNode(-1)
curr = dummy  # curr永远指向最后一个有效节点
```

**Dummy的优势**：
1. 不需要单独处理头节点
2. 代码更简洁、更不容易出错
3. 返回时直接 `dummy.next`

---

## 链表相关扩展题目

### 扩展1：反转链表（LeetCode #206）

```python
def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    反转链表
    时间: O(n), 空间: O(1)
    """
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # 保存下一节点
        curr.next = prev        # 反转指向
        prev = curr             # 移动prev
        curr = next_temp        # 移动curr
    
    return prev  # prev现在是新的头节点
```

### 扩展2：环形链表检测（LeetCode #141）

```python
def has_cycle(head: Optional[ListNode]) -> bool:
    """
    快慢指针检测环
    时间: O(n), 空间: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

---

## 总结

| 技巧 | 适用场景 |
|-----|---------|
| 哈希表 | 需要O(1)查找的场景（去重、查找、两数之和） |
| Dummy节点 | 链表合并、删除节点时简化边界处理 |
| 快慢指针 | 链表环检测、找链表中点 |
| 双指针 | 有序数组合并、滑动窗口 |

**核心心法**：
1. 数组优势：**O(1)随机访问**，适合需要频繁索引的场景
2. 链表优势：**O(1)插入删除**（已知位置），适合动态修改的场景
3. 空间换时间：用哈希表将O(n²)优化到O(n)
