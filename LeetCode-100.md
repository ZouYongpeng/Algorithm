# LeetCode 100 道

## N个数之和

### [两数之和](https://leetcode-cn.com/problems/two-sum/)

> 给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

第一种：暴力，时间复杂度O(N*N)

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n-1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[0];
    }
}
```

第二种：使用HashMap，时间复杂度O(N)
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; ++i) {
            if (hashtable.containsKey(target - nums[i])) {
                return new int[]{hashtable.get(target - nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }
}
```



### [三数之和](https://leetcode-cn.com/problems/3sum/)

> 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
>
> 注意：答案中不可以包含重复的三元组。

```java
public class SumOfThreeNums {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        // 先排序
        Arrays.sort(nums);
        int size = nums.length;
        for (int first = 0; first < size - 2; first++) {
            // 跳过重复
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            int second = first + 1;
            int third = size - 1;
            for (; second < size - 1; second++) {
                // 跳过重复
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 保证 nums[first] < nums[second] < nums[third],
                // 如果三数之和大于0，那么让third往左走
                while (second < third && nums[first] + nums[second] + nums[third] > 0) {
                    third--;
                }
                if (second == third) {
                    break;
                }
                if (nums[first] + nums[second] + nums[third] == 0) {
                    List<Integer> list = new ArrayList<>();
                    // 注意要返回的时对应的数而不是下标
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    lists.add(list);
                }
            }
        }
        return lists;
    }

}
```



### [四数之和](https://leetcode-cn.com/problems/4sum/)

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> lists = new ArrayList<>();
        // 先排序
        Arrays.sort(nums);
        int size = nums.length;
        for (int first = 0; first < size - 3; first++) {
            // 跳过重复
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            for (int second = first + 1; second < size - 2; second++) {
                // 跳过重复
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                int third = second + 1;
                int forth = size - 1;
                for (; third < size - 1; third++) {
                    // 跳过重复
                    if (third > second + 1 && nums[third] == nums[third - 1]) {
                        continue;
                    }
                    // 保证 nums[first] < nums[second] < nums[third] < nums[forth],
                    // 如果四数之和大于0，那么让forth往左走
                    while (third < forth && 
                            nums[first] + nums[second] + nums[third] + nums[forth] > target) {
                        forth--;
                    }
                    if (third == forth) {
                        break;
                    }
                    if (nums[first] + nums[second] + nums[third] + nums[forth] == target) {
                        List<Integer> list = new ArrayList<>();
                        // 注意要返回的时对应的数而不是下标
                        list.add(nums[first]);
                        list.add(nums[second]);
                        list.add(nums[third]);
                        list.add(nums[forth]);
                        lists.add(list);
                    }
                }
            }
        }
        return lists;
    }
}
```



## 滑动窗口

### [无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

> 给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。
>
> ```
> 输入: s = "abcabcbb"
> 输出: 3 
> 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
> ```

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s.length() == 0) {
            return 0;
        }
        int left = 0; // 滑动窗口左侧位置
        int max = 0;  // 滑动窗口最大长度

        // 跳过 hashMap 存储 字符s[i]出现过的位置
        HashMap<Character, Integer> map = new HashMap();
        for(int i = 0; i < s.length(); i++) {
            if(map.containsKey(s.charAt(i))) {
                // 更新滑动窗口
                int newLeft = map.get(s.charAt(i)) + 1;
                left = Math.max(left, newLeft);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }
}
```



## [两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

> 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
>
> 请你将两个数相加，并以相同形式返回一个表示和的链表。
>
> 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
>
> ```
> 输入：l1 = [2,4,3], l2 = [5,6,4]
> 输出：[7,0,8]
> 解释：342 + 465 = 807.
> ```

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null;
        ListNode tail = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            // listNode 为null时，val当作o
            int num_1 = l1 == null ? 0 : l1.val;
            int num_2 = l2 == null ? 0 : l2.val;
            int sum = num_1 + num_2 + carry;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry != 0) {
            tail = new ListNode(carry);
        }
        return head;
    }

}
```

