package leetcode;

import jdk.nashorn.api.tree.Tree;

import javax.imageio.ImageTranscoder;
import java.security.PublicKey;
import java.sql.Array;
import java.sql.SQLIntegrityConstraintViolationException;
import java.util.*;

public class Leetcode {


    public static void main(String[] args){
       HashSet<Integer> hashset = new HashSet<>();
       hashset.add(1);
       hashset.add(2);
       for (int i : hashset){
           System.out.println((int)(i));
       }
    }

    //将给定的数转换为字符串，原则如下：1对应a，2对应b，…，26对应z。 abbeh”,”aveh”,”abyh”,”lbeh”,”lyh”
    //编写函数给出可以转换的字符串的个数。
    public void replace(String str){
        if (str.length() == 0)
            return;

        dfs(str,0,"");
    }

    public void dfs(String s,int index,String result){
        if (index >= s.length()){
            System.out.println(result);
            return;
        }

        int number = s.charAt(index) - '0';
        if (number >= 1 && number <= 26 ) {
            char c = (char) ('a' + number - 1);
            dfs(s, index + 1, result + c);
        }

        if (index + 1 < s.length()){
            int number2 = s.charAt(index+1) - '0';
            int value = number * 10 + number2;
            char c = (char)('a' + value - 1);
            if (c <= 'z' && c>= 'a')
             dfs(s, index + 2, result + c);
        }

    }



    public String replaceSpace(StringBuffer str) {

        int number = 0;
        for (int i = 0; i < str.length(); i++) {
            if (' ' == str.charAt(i))
                number += 2;
        }

        char[] arr = new char[str.length() + number];
        int index = str.length() - 1;
        int index2 = arr.length - 1;
        while (index >= 0) {
            if (str.charAt(index) != ' ') {
                arr[index2--] = str.charAt(index--);
            } else {
                arr[index2--] = '0';
                arr[index2--] = '2';
                arr[index2--] = '%';
                index--;
            }
        }
        return String.valueOf(arr);
    }

    public String reverseWords(String s) {
        while (s.length() > 0 && s.charAt(0) == ' ') {
            char c = s.charAt(0);
            if (c == ' ' && s.length() == 1) {
                s = "";
            } else {
                s = s.substring(1);
            }
        }
        if (s.length() == 0)
            return s;

        StringBuilder stringBuilder = new StringBuilder();
        String[] re = s.split("//s+");
        for (int i = re.length - 1; i > 0; i--) {
            if (re[i].equals(" "))
                continue;
            stringBuilder.append(re[i]);
            stringBuilder.append(" ");
        }
        stringBuilder.append(re[0]);
        return stringBuilder.toString();
    }

    //分割字符串  dfs
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        List<String> list = new ArrayList<>();
        dfs(s, 0, result, list);
        return result;


    }

    public void dfs(String s, int begin, List<List<String>> result, List<String> list) {
        if (begin == s.length()) {
            List<String> list1 = new ArrayList<>();
            for (String string : list)
                list1.add(string);
            result.add(list1);
            return;
        }
        for (int i = begin; i < s.length(); i++) {
            String t = s.substring(begin, i + 1);
            if (isPartition(t)) {
                list.add(t);

                dfs(s, i + 1, result, list);

                list.remove(list.size() - 1);
            }
        }
    }

    public boolean isPartition(String s) {
        int begin = 0;
        int end = s.length() - 1;

        while (end > begin) {
            if (s.charAt(end) != s.charAt(begin))
                return false;
            else {
                begin++;
                end--;
            }
        }
        return true;
    }

    HashMap<String, List<String>> hashMap = new HashMap<>();
    public List<String> wordBreak2(String s, List<String> wordDict) {
        if (hashMap.containsKey(s)) {
            return hashMap.get(s);
        }
        List<String> list = new ArrayList<>();
        if (0 == s.length()) {
            list.add("");
            return list;
        }
        for (String str : wordDict) {
            if (s.startsWith(str)) {
                List<String> subList = wordBreak2(s.substring(str.length()), wordDict);
                for (String sub : subList) {
                    list.add(str + (Objects.equals("", sub) ? "" : " ") + sub);
                }
            }
        }
        hashMap.put(s, list);
        return list;
    }

    public int firstUniqChar(String s) {
        int[] hashmap = new int[128];
        for (int i = 0; i < s.length(); i++) {
            hashmap[s.charAt(i)]++;
        }

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (hashmap[c] == 1)
                return i;
        }
        return -1;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode index1 = l1;
        ListNode index2 = l2;
        ListNode result = new ListNode(0);

        ListNode pre = result;
        int ans = 0;
        while (index1 != null && index2 != null) {
            int sum = index1.val + index2.val + ans;
            ans = sum / 10;
            sum = sum % 10;
            ListNode t = new ListNode(sum);
            pre.next = t;
            pre = pre.next;
            index1 = index1.next;
            index2 = index2.next;
        }

        while (index1 != null) {
            int sum = index1.val + ans;
            ans = sum / 10;
            sum = sum % 10;
            ListNode t = new ListNode(sum);
            pre.next = t;
            pre = pre.next;
            index1 = index1.next;
        }
        while (index2 != null) {
            int sum = index2.val + ans;
            ans = sum / 10;
            sum = sum % 10;
            ListNode t = new ListNode(sum);
            pre.next = t;
            pre = pre.next;
            index2 = index2.next;
        }
        if (ans != 0) {
            ListNode t = new ListNode(ans);
            pre.next = t;
        }
        return result.next;
    }

    public int lengthOfLongestSubstring(String s) {
        int[] m = new int[128];
        Arrays.fill(m, -1);
        int res = 0, left = -1;
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            left = Math.max(left, m[c]); //第一次出现  left不变   第二次出现 left 会记录下来第二次出现字母坐标 并设为七点
            m[c] = i;
            res = Math.max(res, i - left);
        }
        return res;
    }

    public String longestPalindrome(String s) {
        if (s.isEmpty()) return "";
        int[][] dp = new int[s.length()][s.length()];
        int left = 0, right = 0, len = 0;
        for (int i = 0; i < s.length(); ++i) {
            for (int j = 0; j < i; ++j) {
                dp[j][i] = (s.charAt(i) == s.charAt(j) && (i - j < 2 || dp[j + 1][i - 1] > 0)) ? 1 : 0;
                if (dp[j][i] > 0 && len < i - j + 1) {
                    //     len = i - j + 1;
                    left = j;
                    right = i;
                }
            }
            dp[i][i] = 1;
        }
        return s.substring(left, right + 1);
    }

    public List<List<Integer>> threeSum(int[] nums) {
        /**求三数之和为0的，且不需要原始索引，那么可以先排序，然后两边缩减一一尝试判断即可
         三数和为0，那就是两个小的值加一个大的值，固定一个改变其他两个遍历数组就可以
         **/
        List<List<Integer>> result = new ArrayList<>();
        //先排序
        Arrays.sort(nums);
        //遍历数组，逐一尝试，三个数，则i只需要到length - 3即可；
        for(int i = 0; i < nums.length - 2; i++){
            int l = i + 1;
            int r = nums.length - 1;
            if (nums[i] > 0) break;

            if (i>0 &&  nums[i] == nums[i-1]) continue;

            while (r>l){
                int sum = nums[i] + nums[l] + nums[r];

                if (sum == 0){
                    List<Integer> list = Arrays.asList(nums[i],nums[l],nums[r]);
                    result.add(list);

                    //去重
                    while (r > l && nums[l] == nums[l+1]) l++;
                    while (r > l && nums[r] == nums[r-1]) r--;
                    r--;
                    l++;
                }else if (sum > 0)
                    r--;
                else
                    l++;

            }

        }
        return result;
    }

    //电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        List<String> map = new ArrayList<>();

        if (digits.length() == 0)
            return result;

        map.add("");
        map.add("");
        map.add("abc");
        map.add("def");
        map.add("ghi");
        map.add("jkl");
        map.add("mno");
        map.add("pqrs");
        map.add("tuv");
        map.add("wxyz");
        StringBuilder tmp = new StringBuilder();
        dfs(0,digits,result,map,tmp);
        return result;
    }

    public void dfs(int index, String digits,List<String> result,List<String> map,StringBuilder tmp){
        if (index == digits.length() - 1)
        {
            String t = tmp.toString();
            result.add(t);
            return;
        }

       // for (int i = index; i < digits.length(); i++){
            int num = digits.charAt(index) - '0';
            for (int j = 0; j < map.get(num).length(); j++){
                tmp.append(map.get(num).charAt(j));
                dfs(index+1,digits,result,map,tmp);
                tmp.deleteCharAt(tmp.length()-1);
            }
        //}

    }


    //Remove Nth Node From End of List
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null || head.next == null)
            return null;

        ListNode p = head;
        int len = lengthListNode(head);
        n = len - n + 1;
        if (n == len){
            len = len - 2;
            while (len != 0){
                p = p.next;
                len--;
            }
            p.next = null;
        }else if (n == 1){
            p = p.next;
            head.next = null;
            head = p;
        }else {
            n = n -2;
            while (n != 0){
                p = p.next;
                n--;
            }
            p.next = p.next.next;
        }
        return head;

    }

    public int lengthListNode (ListNode listNode){
        int ans = 0;
        while (listNode != null){
            listNode = listNode.next;
            ans++;
        }
        return ans;
    }

    //Valid Parentheses
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        Map<Character,Character> map = new HashMap<>();
        map.put(']','[');
        map.put(')','(');
        map.put('}','{');

        for (int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if (c == '[' || c == '(' || c=='{')
                stack.push(c);
            else {
                char t = map.get(c);

                if (stack.size() == 0 || stack.peek() != t)
                    return false;
                else
                    stack.pop();
            }
        }
        if (stack.size() == 0)
            return true;
        else
            return false;
    }


    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        Arrays.sort(candidates);//排序
        dfs(0,result,target,candidates,list);
        return result;
    }

    public void dfs(int index,List<List<Integer>> result , int target, int[] candidates,ArrayList<Integer> tmp){
        if (target == 0){
            List<Integer> t = (List<Integer>) tmp.clone();
            result.add(t);
            return;
        }else if (target < 0){
            return;
        }

        for (int i = index; i < candidates.length; i++){
            tmp.add(candidates[i]);
            dfs(i,result,target-candidates[i],candidates,tmp);
            tmp.remove(tmp.size()-1);
        }
    }

    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int i = n - 2;
        int j = n - 1;
        while (i >= 0 && nums[i] >= nums[i + 1]) --i;
        if (i >= 0) {
            while (nums[j] <= nums[i]) --j;
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }
        Arrays.sort(nums,i+1,nums.length-1);
    }


    public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> tmp = new ArrayList<>();
    Arrays.sort(nums);
    permuteDFS(result,0,nums);
    return result;
}

    public void permuteDFS(List<List<Integer>> result,int index,int[] nums){
    System.out.println("nums : len  " + nums.length);
        if (index == nums.length) {
            ArrayList arrayList = new ArrayList();
            for (int i = 0; i < nums.length; i++){
                arrayList.add(nums[i]);
                System.out.println(nums[i]);
            }
            result.add(arrayList);
            return;
        }

        for (int i = index; i < nums.length; ++i) {
            int j = i - 1;
            while (j >= index && nums[j] != nums[i]) --j;
            if (j != index - 1) continue;
            swapPermute(nums, i,index);
            permuteDFS(result, index + 1, nums);
            swapPermute(nums,i,index);
        }

    }

    public void swapPermute(int[] nums,int index1,int index2){
        int tmp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = tmp;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();
        Map<String,List<String>> hashMap = new HashMap<>();

        for (String str : strs){
            char[] s = str.toCharArray();
            Arrays.sort(s);
            String tmp = new String(s);
            if (!hashMap.containsKey(tmp))
                hashMap.put(tmp,new ArrayList<String>());
            hashMap.get(tmp).add(str);
        }

        for (List<String> list : hashMap.values()){
            result.add(list);
        }
        return result;

    }


    public int maxSubArray(int[] nums) {

        if (nums.length == 0 || nums == null)
        {
            System.out.println("nums is null or 0");
            return -1;
        }
        //dp
        int result = Integer.MIN_VALUE;

        int pre  = nums[0];
        for (int i = 0; i < nums.length;i++){
            int t = pre + nums[i];
            int compareMax = t > nums[i] ? t: nums[i];
            pre = compareMax;
            result = Math.max(compareMax,result);
        }
        return result;
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < matrix.length; i++){
            for (int j = i+1; j < matrix[0].length; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
            //reverse
            for (int j = 0; j < matrix[i].length / 2; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix.length-1-j];
                matrix[i][matrix.length-1-j] = tmp;
            }
        }
    }

    public boolean canJump(int[] nums) {

        int n = nums.length;
        int result = 0;
        for (int i = 0; i < n; i++){
            if (result > n || i > result) break;
            result = Math.max(result,nums[i] + i);

        }

        return result >= n - 1;
    }

    //Merge Intervals   合并区间   注意用指针来理解 别用c++思想
    public List<Interval> merge(List<Interval> intervals) {
            Collections.sort(intervals, new Comparator<Interval>() {
                @Override
                public int compare(Interval o1, Interval o2) {
                    return o1.start - o2.start;
                }
            });

            List<Interval> result = new ArrayList<>();
            Interval pre = null;
            for (Interval cur : intervals){
                if (pre == null || cur.start > pre.end){
                    result.add(cur);
                    pre = cur;
                }else if (cur.end > pre.end){
                    pre.end =cur.end;  //指向对象，直接修改结果集
                }
            }

            return  result;
    }

    //dp   寻找最小路径  可以使用dfs
    public int minPathSum(int[][] grid) {
        if (grid == null|| grid.length == 0 || grid[0].length == 0)
            return 0;

        int[][] dp = new int[grid.length][grid[0].length];
        int col = grid.length;
        int row = grid[0].length;
        dp[0][0] = grid[0][0];
        for (int i = 1; i < col; i++)
            dp[i][0] = dp[i-1][0] + grid[i][0];

        for (int i = 1; i < row; i++)
            dp[0][i] = dp[0][i-1] + grid[0][i];

        for (int i = 1 ;i < col; i++){
            for (int j = 1; j < row; i++){
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[col-1][row-1];
    }

    //Climbing Stairs  简单dp
    public int climbStairs(int n) {
        if (n == 1 ) return 1;
        if(n == 2) return 2;

        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 2;

        for (int i = 3; i <= n; i++){
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }

        //二分  Find First and Last Position of Element in Sorted Array
            public int[] searchRange(int[] nums, int target) {
                if (nums ==null || nums.length == 0)
                    return new int[]{-1,-1};


                int[] result = new int[2];

                int left = 0;
                int right = nums.length - 1;
                while (left < right){
                    int mid = left + (right -left) / 2;
                    if (nums[mid]  < target) left = mid + 1;
                    else right = mid;
                }
            if (nums[left] != target)
                return new int[]{-1,-1};


            int index1 = left;
            for (int i = left; i < nums.length; i++){
                if (target == nums[i]) index1 = i;
                else break;
            }

            int index2 = left;
            for (int i = left; i >= 0; i--){
                if (target == nums[i]) index2 = i;
                else break;
            }
            result [0] = index2; result[1] = index1;
            return result;
        }

    //Search in Rotated Sorted Array
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return -1;
        int left = 0;
        int right = nums.length -1;

        while (left < right){
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;

            if (nums[mid] < nums[right] ){
                if (nums[mid] < target &&  nums[right] >=target) left = mid + 1;
                else right = mid;
            }else if (nums[mid] > nums[right]){
                if (nums[left] <= target && nums[mid] >target) right = mid ;
                else left =mid + 1;
            }

        }

        if(nums[left] == target)
            return left;
        else
        return -1;
    }

    //Sort Colors  双指针法 注意左边是确定顺序   右边具有不确定性  注意循环的边界问题
    public void sortColors(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        for (int i = 0; i <= right; i++){
            if (nums[i] == 0){
                swap(nums,i,left);

                left++;
            }else if(nums[i] == 2){
                swap(nums,i,right);
                right--;
                i--;
            }
        }


    }
    public void swap(int[] nums, int index1,int index2){
        int tmp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = tmp;
    }

    //Subsets  找出子集的问题  可以使用动态规划   遍历 在原有的基础之上使用 构建子集
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        result.add(tmp);

        for (int i = 0; i < nums.length; i++){
          //  System.out.println("i"  + i);
            int n = result.size();
            for (int j = 0; j < n; j++){

            //    System.out.println("result"  + result.size());
                List<Integer> t = new ArrayList<>(result.get(j));
                t.add(nums[i]);
                result.add(t);
            }
        }
        return result;
    }

    //dfs  搜索二维数组内部 是否有目标单词
    public boolean exist(char[][] board, String word) {
        if (word.length() == 0)
            return true;
        if (board == null || board.length == 0 || board[0].length == 0)
            return false;
        boolean[][] visit = new boolean[board.length][board[0].length];


        for (int i = 0; i < board.length; i++){
            for (int j = 0; j < board[0].length; j++){
                if (dfs(board,word,0,visit,i,j))
                    return true;
            }
        }
        return false;
    }

    public boolean dfs(char[][] board,String word,int index,boolean[][] visit,int col,int row){


        if (col >= board.length || col<0 || row >= board[0].length || row < 0 || index >= word.length()||
        visit[col][row] == true )
            return false;

        if (word.charAt(index) != board[col][row])
            return false;
        //
        if (index == word.length() - 1)
        {
          return true;
        }

        visit[col][row] = true;

        boolean result = dfs(board,word,index+1,visit,col+1,row) ||
                dfs(board,word,index+1,visit,col-1,row) ||
                dfs(board,word,index+1,visit,col,row+1) ||
                dfs(board,word,index+1,visit,col,row-1);

        visit[col][row] = false;
        return result;
    }


    //中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null)
            return result;
        inorder(result,root);
        return result;
    }
    public void inorder(List<Integer> result, TreeNode root){
        if (root.left != null)
            inorder(result,root.left);

        result.add(root.val);
        if (root.right != null)
            inorder(result,root.right);
    }


    //Unique Binary Search Trees 给结点数 求可能构成avl树的可能性个数  dp  左子树和右子树的相乘为其可能性
    public int numTrees(int n) {
            if (n == 0 || n == 1) return 1;
            int[] dp = new int[n+1];
            dp[0] =1;
            dp[1] = 1;
            for (int i = 2; i <= n; i++){
                for (int j = 0; j < i; j++){
                    dp[i] += dp[j] * dp[i-j-1];
                }
            }
            return dp[n];
    }

    //判断是否为二叉搜索树  Validate Binary Search Tree  中序遍历为一个递增数组
    public boolean isValidBST(TreeNode root) {
      if (root == null)
          return true;
      List<Integer> result = new ArrayList<>();
      inorder(result,root);
      for (int i = 0; i < result.size() -1; i++){
          if (result.get(i) > result.get(i+1)) return false;
      }
      return true;
    }

    // Symmetric Tree  判断镜像树
    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;


        return dfsSymmetric(root.left , root.right);


    }

    public boolean dfsSymmetric(TreeNode x,TreeNode y){
        if (x == null && y == null)
            return true;
        if (x == null  && y != null || x != null && y == null)
            return false;

        if (x.val != y.val)
            return false;

        return dfsSymmetric(x.left,y.right) && dfsSymmetric(x.right,y.left);
    }


    // Binary Tree Level Order Traversal  二叉树的层次遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null)
            return result;

        LinkedList<TreeNode> cur = new LinkedList<>();
        //LinkedList<TreeNode> pre = new LinkedList<>();

        cur.add(root);
        ArrayList<Integer> level = new ArrayList<>();
       // level.add(root.val);
      //  result.add((List<Integer>) level.clone());
        while (!cur.isEmpty()){
            int len = cur.size();
            level.clear();
            for (int i = 0; i < len ;i++){
                TreeNode tmp = cur.getFirst();
                level.add(tmp.val);
                cur.removeFirst();

                if (tmp.left != null){
                    cur.add(tmp.left);
                }
                if (tmp.right != null){
                    cur.add(tmp.right);
                }
            }
            result.add((List<Integer>) level.clone());
        }


        return result;

    }


    //Maximum Depth of Binary Tree
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;

        int left = maxDepth(root.left);
        int right = maxDepth(root.right);

        return left>right ? left + 1 :right + 1;

    }


    //Best Time to Buy and Sell Stock
    public int maxProfit(int[] prices) {
        int result = 0;
        if (prices == null || prices.length == 0)
            return result;

        int left = prices[0];

        for (int i = 1 ; i < prices.length; i++){
            if (prices[i] < left)
                left = prices[i];
            else {
                result = Math.max(prices[i] - left,result);
            }
        }
        return result;
    }


    //知道前序中序遍历  还原二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || preorder.length == 0)
            return null;

        return dfsBulidTree(preorder,inorder,0,preorder.length,0,inorder.length);

    }

    public TreeNode dfsBulidTree(int[] preorder, int[] inorder,int beginPre,int endPre,int beginIno, int endIno){
        if (beginIno == endIno)
            return null;

        //左闭右开
        int rootValue = preorder[beginPre];
        TreeNode root = new TreeNode(rootValue);

        int i;
        for ( i = beginIno; i < endIno; i++){
            if (inorder[i] == rootValue)
                break;
        }

        root.left = dfsBulidTree(preorder,inorder,beginPre + 1,beginPre + 1+ (i-beginIno),
                beginIno,i);

        root.right = dfsBulidTree(preorder,inorder, beginPre + (i - beginIno) +1 , endPre,i+1,endIno);

        return root;

    }

    //. Flatten Binary Tree to Linked List  递归法 递归可以确定左右子树成为链表状态  我们负责针对一节点平铺为链表
    public void flatten(TreeNode root) {
        if (root == null) return;
        if (root.left!=null) flatten(root.left);
        if (root.right!=null) flatten(root.right);
        TreeNode tmp = root.right;
        root.right = root.left;
        root.left = null;
        while (root.right != null) root = root.right;
        root.right = tmp;

    }


    //Single Number     位的异或运算  满足交换律
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int result = nums[0];
        for (int i = 1; i < nums.length; i++)
            result = result ^ nums[i];

        return result;

    }

    //139. Word Break  dp bfs dfs 这里使用dp
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> hashSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;

        //dp【i】 定义的是从0到i的左闭右开的区间  dp【0】为空  dp【1】 为长度为1 的字符串 s

        for (int i = 1; i < s.length()+1; i++){
            for (int j = 0; j < i; j++)
            {
                if (dp[j] == true && hashSet.contains(s.substring(j,i))) {
                    //   System.out.println("  :" + i);
                    dp[i] = true;
                    break;
                }
            }
        }
        // for (int i = 0; i < s.length() + 1; i++)
        //   System.out.println(" dp [i] " + i + "  " + dp[i]);
        return dp[s.length()];
    }


    //链表是否成环  Linked List Cycle
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null)
            return false;

        ListNode fast = head;
        ListNode slow = head;

        while (fast != null && fast.next != null && slow != null){
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast)
                return true;
        }
        return false;
    }

    //Linked List Cycle II  找出链表成环的入口
    public ListNode detectCycle(ListNode head) {
        if(head == null && head.next == null)
            return null;

        ListNode fast = head;
        ListNode slow = head;

        while(fast != null && fast.next != null && slow != null){
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow)
                break;
        }

        if (fast == null || slow == null || fast.next == null)
            return null;

        fast = head;
        while (fast != slow){
            fast = fast.next;
            slow =slow.next;
        }
        return slow;

    }



    // Maximum Product Subarray
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;

        int[] maxDp = new int[nums.length];
        int[] minDp = new int[nums.length];

        maxDp[0] = nums[0];
        minDp[0] = nums[0];
        int result = nums[0];

        for (int i = 1; i < nums.length; i++){
            int minS = minDp[i-1] * nums[i];

            int maxS = maxDp[i-1] * nums[i];
            result = Math.max(result, Math.max(nums[i],Math.max(minS,maxS)) );

            maxDp[i] = Math.max(nums[i],Math.max(minS,maxS));
            minDp[i] = Math.min(nums[i],Math.min(minS,maxS));
        }
        return result;

    }

    //链表的归并排序   对链表的使用  n log n 的时间复杂度进行排序   归并排序
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode slow = head, fast = head, pre = head;

        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        pre.next = null;
        return merge(sortList(head), sortList(slow));
    }

    public ListNode merge(ListNode l1,ListNode l2){
        if (l1 == null) return l2;
        if (l2 == null )return  l1;
        if (l1.val < l2.val){
            l1.next = merge(l1.next,l2);
            return l1;
        }else {
            l2.next = merge(l1, l2.next);
            return l2;
        }
    }

    //160. Intersection of Two Linked Lists  找出两个链表的交汇点
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null )
            return null;

        int len1 = 0;
        int len2 = 0;
        ListNode p = headA;
        while (p != null){
            len1++;
            p = p.next;
        }

        p = headB;
        while (p != null){
            len2++;
            p = p.next;
        }

        System.out.println(len1 + " len1");
        System.out.println(len2 + " len 2");
        ListNode q = headB;
        if (len1 < len2){
           p = headB;
           q = headA;
        }else {
            p = headA;
            q = headB;
        }
        // A 长
        int run = Math.abs(len1 - len2);

        while (run != 0){
            p = p.next;
            run--;
        }

        while (p != q){
            System.out.println(p.val);
            p = p.next;
            q = q.next;
        }

        return q;
    }

    //169. Majority Element  找出重复出现次数超过数组长度一半的数
    public int majorityElement(int[] nums) {
        if (nums == null || nums.length ==0)
            return 0;

        int result = nums[0];
        int cnt = 1;

        for (int i = 1; i < nums.length; i++){
            if (cnt == 0){
                result = nums[i];
            }

            if (result == nums[i])
                cnt++;
            else {
                cnt--;
            }

        }
        return result;

    }

    //206. Reverse Linked List
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null)
            return head;

        ListNode pre = head;
        ListNode cur = head.next;
        ListNode next = head.next.next;

        pre.next = null;
        while (next != null){
            cur.next = pre;
            pre = cur;
            cur = next;
            next = next.next;
        }
        cur.next = pre;

        return cur;
    }

    //dfs  200. Number of Islands  求岛屿个数
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0|| grid[0].length == 0)
            return 0;

        boolean[][] visit = new boolean[grid.length][grid[0].length];

        int result = 0;
        for (int i = 0; i < grid.length; i++){
            for (int j = 0; j < grid[i].length; i++){
                if (visit[i][j] != true  && grid[i][j] == '1'){
                    dfsNumIslands(grid,i,j,visit);
                    result++;
                }

            }
        }
        return  result;
    }

    public void dfsNumIslands(char[][] grid,int i,int j, boolean[][] visit){
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length)
            return;
        if (visit[i][j] == true || grid[i][j] == '0')
            return;
        visit[i][j] = true;
        dfsNumIslands(grid,i+1,j,visit);
        dfsNumIslands(grid,i-1,j,visit);
        dfsNumIslands(grid,i,j+1,visit);
        dfsNumIslands(grid,i,j-1,visit);
    }





    //198. House Robber  dfs 超时
    //尝试使用dp   dp【i】 = dp[i - 1]  > dp[i - 2] + nums[i]? ---:--- Accepted
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;

        if (nums.length == 1)
            return nums[0];
        if (nums.length == 2){
            return nums[0] > nums[1] ? nums[0] : nums[1];
        }

        int dp[] = new int[nums.length];
        dp[0] = nums[0];
        dp[1] =  nums[0] > nums[1] ? nums[0] : nums[1];

        for (int i = 2; i < nums.length; i++){
            dp[i] = Math.max(dp[i-2] + nums[i] , dp[i-1]);
        }

        return dp[nums.length - 1];
    }


    /*
     private static int   resultRot = 0;
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;

        dfsRob(nums,0,0);
        return  resultRot;
    }

    public void dfsRob(int[] nums, int index, int money){
        if (index >= nums.length){
            resultRot = Math.max(resultRot,money);
            return;
        }
   //     int tmpMoney = money + nums[index];
        dfsRob(nums,index+1,money);
        dfsRob(nums,index+2,money+nums[index]);
    }
    */


    //207 Course Schedule  知识点  有向图  可BFS  也可DFS
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // dfs
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        int[] visit = new int[numCourses];
        for (int i = 0; i < numCourses; i++)
            graph.add(new ArrayList<Integer>());

        for (int i = 0; i < prerequisites.length; i++){
            graph.get(prerequisites[i][1]).add(prerequisites[i][0]);  //我的入度
        }

        for (int i = 0; i < numCourses; i++){
            if (!dfsCanFinish(graph,visit,i)) return false;
        }
        return true;





        //BFS 的写法
       /* if (numCourses == 0 || numCourses == 1)
            return true;

        //记录出度集合
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++)
            graph.add(new ArrayList<Integer>());
        int[] in = new int[numCourses];
        LinkedList<ArrayList<Integer>> listQueue = new LinkedList<>();



        for (int i = 0; i < prerequisites.length; i++){
            //System.out.println("i  " + i);
            //System.out.println(graph[0].size() + " size");
                graph.get(prerequisites[i][0]).add(prerequisites[i][1]);
                in[prerequisites[i][1]]++;
        }
        for (int i = 0; i < numCourses; i++){
            if (in[i] == 0){
                    listQueue.addLast(graph.get(i));
            }
        }
        while (!listQueue.isEmpty()){
            ArrayList<Integer> tmp =listQueue.getFirst();
            listQueue.removeFirst();
            for (int i : tmp){
                in[i]--;
                if (in[i] == 0)
                    listQueue.add(graph.get(i));
            }

        }
        for (int i = 0; i < numCourses; i++){
            if (in[i] > 0)
                return false;
        }
        return true;
        */
    }

    public boolean dfsCanFinish(ArrayList<ArrayList<Integer>> graph, int[] visit,int index){
        if (visit[index] == 1) return true;
        if (visit[index] == -1) return  false;

        visit[index] = -1;

        for (int i = 0; i < graph.get(index).size(); i++){
            if (! dfsCanFinish(graph,visit,graph.get(index).get(i))) return false;
        }
        visit[index] = 1;
        return true;
    }


    //234. Palindrome Linked List  判断会问链表
        public boolean isPalindrome(ListNode head) {
            ListNode fast = head, slow = head;
            while(fast != null && fast.next != null) {
                fast = fast.next.next;
                slow = slow.next;
            }
            // 奇数个结点
            if(fast != null) {
                slow = slow.next;
            }
            // 右半侧反向
            ListNode right = reverse(slow);
            ListNode left = head;
            ListNode helper = right;
            while(right != null) {
                if(left.val != right.val) {
                    return false;
                }
                //left的最后一个节点（即中间点的前一个节点），仍然指向中间点
                //因此奇数情况下，最后一次循环left与right都指向中间点
                left = left.next;
                right = right.next;
            }
            // 恢复右半侧
            reverse(helper);
            return true;
        }

        public ListNode reverse(ListNode head){
            if (head == null || head.next == null) return head;
            ListNode pre =head;
            head = head.next;
            ListNode next = head.next;
            pre.next = null;
            while (next != null){
                head.next = pre;
                pre = head;
                head = next;
                next = next.next;
            }
            head.next = pre;
            return head;
        }



    //226. Invert Binary Tree  翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return  null;
        TreeNode left = root.left;
        TreeNode right = root.right;
        root.left = invertTree(right);
        root.right = invertTree(left);
        return root;
    }

    //238. Product of Array Except Self
    public int[] productExceptSelf(int[] nums) {
        int[] result = new int[nums.length];
        int len = nums.length;
        int right = 1;
        result[0] = 1;
        for (int i = 1; i < len; i++){
            result[i] = result[i-1] * nums[i-1];  //result[i] 记录额使用 [0,i] 的左闭右开区间
        }
        for (int i = len - 2; i >= 0; i--){
            right = right * nums[i+1];
            result[i] = result[i-1] * right;
        }

        return result;
    }

    //215. Kth Largest Element in an Array  快排题
    public int findKthLargest(int[] nums, int k) {

        quikSort(nums,0,nums.length - 1,nums.length  -k);
        return nums[nums.length  -k];

    }
    public int quikSort(int[] nums,int  begin, int end,int k){  //左闭右开
        if (end > begin){
            int index = qSort(nums,begin,end);
            if (index == k)
                return 0;
            quikSort(nums,begin,index-1,k);
            quikSort(nums,index+1,end,k);
        }
        return 0;
    }

    public int qSort(int[] nums,int begin, int end){

        int value = nums[begin];

        while (begin < end) {
            while (begin < end && nums[end] >= value)
                end--;
            if (begin < end)
                nums[begin++] = nums[end];
            while (begin < end && nums[begin] <= value)
                begin++;
            if (begin < end)
                nums[end--] = nums[begin];
        }
        nums[begin] = value;
        return begin;
    }

    //236. Lowest Common Ancestor of a Binary Tree  寻找子节点最小的共同祖先   递归思想
    /**
     * 分三种情况
     * 1 都在左子树
     * 2 都在右子树
     * 3 在一左一右
     *
     *
     * 设计 找到其中之一返回其指针位置   这个位置如果是两个结点的共同祖先 就替代返回  如果不是 就直接返回
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == q || root == p) return root;

        TreeNode left = lowestCommonAncestor(root.left,p,q);
        if (left != null &&  left != q && left != p )return  left;

        TreeNode right = lowestCommonAncestor(root.right, p ,q);
        if (right != null && right != q && right!=p) return right;

        if (left != null && right!= null)
            return root;

        return left != null ? left : right;

    }

    //221. Maximal Square  找出图中最大1的正方形  dp   dp[i][j]=Math.min(Math.min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1;
    //类似于机器人走路   当前位置记录的是以这个点为右下角的正方形
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;

        int col = matrix.length;
        int row = matrix[0].length;
        int[][] dp = new int[matrix.length][matrix[0].length];

        int result = 0;
        for(int i = 0; i < col; i++){
            if (matrix[i][0] == '1'){
                result = 0 ;
                dp[i][0] = 1;
            }
        }
        for (int i = 0; i < row; i++){
            if (matrix[0][i] == '1'){
                result = 1;
                dp[0][i] = 1;
            }
        }
        for (int i = 1; i < col; i++){
            for(int j = 1; j < row; j++){
                if (matrix[i][j] == '1'){
                    dp[i][j] = Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1])) + 1;
                    result = result > dp[i][j] ? result : dp[i][j];
                }
            }
        }
        return  result * result;

    }


    //240. Search a 2D Matrix II  判断大小关系  起始点从右上  或者左下进行搜索
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix == null ||  matrix.length == 0 || matrix[0].length == 0){
            return false;
        }

        int x = 0;
        int y = matrix[0].length - 1;
        while (x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length){
            if (matrix[x][y] > target){
                y--;
            }else if (matrix[x][y] < target){
                x++;
            }else {
                return true;
            }
        }

        return false;
    }


    //279. Perfect Squares  找出最少个数   平方数之和  尝试贪心 不对  使用dp 一直向前推导
    public int numSquares(int n) {
        int dp[] = new int[n+1];
        for (int i =0; i <=n ;i++)
            dp[i] = Integer.MAX_VALUE;
        dp[0] = 0;

        for (int i = 1; i <= n; i++){
            for (int j = 1; j*j + i < n; j++){
                dp[i + j * j] = Math.min(dp[i + j * j], dp[i] + 1);
            }
        }

        return dp[n];

    }
    //283. Move Zeroes
    public void moveZeroes(int[] nums) {
        for (int i = 0, j = 0; i < nums.length;i++ ){
            if (nums[i] != 0){
                swap(nums,i,j++);
            }
        }
    }

    //287. Find the Duplicate Number 不用多余空间  不能改变数组
    public int findDuplicate(int[] nums) {
        int left = 0;
        int right = nums.length;

        while (left < right){
            int mid = left + (right - left) / 2;
            int cnt = 0;
            for (int tmp : nums){
                if (tmp <= mid)
                    cnt++;
            }

            if (cnt <= mid)  left = mid + 1;
            else right = mid;

        }
        return right;
    }

    class LISnote{
       public int number;
       public int value;

    }

    //300. Longest Increasing Subsequence  dp中的lis   最长上升序列
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;

        int[] low = new int[nums.length + 1];

        for (int i = 1; i <= low.length; i++)
            low[i] = Integer.MAX_VALUE;

        low[1] = nums[0];
        int ans = 1;
        for (int i = 1; i < nums.length; i++){
            if (nums[i] >= low[ans]){
                low[++ans] = nums[i];
            }else {
                low[binary_search(low,ans+1,nums[i])] = nums[i];
            }
        }
        return ans;
        /*
        int[] dp = new int[nums.length];
        int result = 1;
        for (int i = 0; i < dp.length; i++)
            dp[i] = 1;

        for (int i = 0;i < dp.length; i++){
            for (int j = 0; j < i; j++){
                if(nums[i] > nums[j])
                    dp[i] = Math.max(dp[i],dp[j] + 1);
            }
        }
        for(int i = 0; i < nums.length; i++)
            result = Math.max(result,dp[i]);
        return  result;
        */
        }

        public int binary_search(int[] low,int right,int value){
            int left = 1;
            while (left < right){
                int mid = left + (right - left) / 2;
                if (low[mid] < value) left = mid+1;
                else right = mid;
            }
            return left;
        }

        //309. Best Time to Buy and Sell Stock with Cooldown  dp  画出状态图  三状态可以使用三个数组来维护
        public int maxProfit2(int[] prices) {
            if(prices == null || prices.length <= 1)
                return 0;

            int[] s0 = new int[prices.length];
            int[] s1 = new int[prices.length];
            int[] s2 = new int[prices.length];

            s0[0] = 0;
            s1[0] = -prices[0];
            for(int i = 1; i < prices.length; i++){
                s0[i] = Math.max(s0[i-1],s2[i-1]);
                s1[i] = Math.max(s1[i-1],s0[i-1] - prices[i]);
                s2[i] = s1[i-1] + prices[i];

            }

            return Math.max(s0[prices.length - 1], s2[prices.length - 1]);
        }

    //322. Coin Change   dp  dp[i] 表示 金额为i时最小的硬币数
    public int coinChange(int[] coins, int amount) {
        if(coins == null || coins.length == 0 ) return -1;
        if(amount <1) return 0;

        int[] dp = new int[amount+1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;

        for(int coin: coins){   //如果从硬币的金额开始递归  这样回避免一些溢出问题 提升一些效率
            for(int i=coin;i<=amount;i++){
                if(dp[i-coin]!=Integer.MAX_VALUE){
                    dp[i] = Math.min(dp[i],dp[i-coin]+1);
                }
            }
        }

        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }

    //337. House Robber III
    /**
     * 思路  直接在树的dfs之中用dp  当前节点的最优解为两种情况   dp + dfs
     * 1 这一层不取
     * rootMoney.notUse = 左子节点的最大值+右子节点的最大值（无所谓左子节点和右子节点是否使用）
     * rootMoney.use = root.val + 左子节点的notUse + 右子节点的NotUse。
     */

    // 第二种解法
    class Money{
        public int notUse;
        public int use;

        public Money(){
            notUse = 0;
            use = 0;
        }
    }

    public int rob(TreeNode root) {
        Money money = getLargestMoney(root);

        return Math.max(money.notUse,money.use);
    }

    public Money getLargestMoney(TreeNode root){
        if (root == null)
            return  new Money();

        Money left = getLargestMoney(root.left);
        Money right = getLargestMoney(root.right);

        Money money = new Money();
        money.use = root.val + left.notUse + right.notUse;
        money.notUse = Math.max(left.notUse,left.use) + Math.max(right.use,right.notUse);
        return money;
    }


    //338. Counting Bits
    /**
     * 奇数比偶数多加一个  1
     * 偶数的位数是他的 1/2的一样  只不过是向右移了一位
     */
    public int[] countBits(int num) {
        int result[] = new int[num+1];

        result[0] = 0;
        if (num == 0)
            return result;
        result[1] = 1;
        for(int i = 2; i <= num; i++){
           if (i % 2 == 0){
               result[i] = result[i>>2];
           }else{
               result[i] = 1 + result[i-1];
           }
        }
        return result;
    }


    //347. Top K Frequent Elements
    //桶排序  先用hashmap先遍历一遍   在new 一个nums长度的ArrayList  把key值和value值互换 value代表pinlv
    //最后从频率最高来返回结果
    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer,Integer> hashMap = new HashMap<>();
        for (int value : nums){
            if (!hashMap.containsKey(value))
                hashMap.put(value,1);
            else
                hashMap.replace(value,hashMap.get(value)+1);
        }
        ArrayList<Integer>[] buckets = new ArrayList[nums.length+1];

        for(int key : hashMap.keySet()){
            int frequen = hashMap.get(key);
            if (buckets[frequen] == null) {
                buckets[frequen] = new ArrayList<>();
                buckets[frequen].add(key);
            } else {
                buckets[frequen].add(key);
            }
        }
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = nums.length; i>= 0 && result.size() < k; i--){
            if (buckets[i] == null)
                continue;
            result.addAll(buckets[i]);
        }
        return result;
    }

    //406. Queue Reconstruction by Height
    //一个排序和插入的过程，按照身高进行降序排序，然后把身高相同的人按照k进行升序排序。每次取出身高相同的一组人，按照k值把他们插入到队列中。
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int []>() {        //按身高降序排序(h大的在前面)，按k的大小升序排列(k小的在前面)
            public int compare(int[] a, int[] b) {
                if(a[0] != b[0]) return -a[0]+b[0];
                else return a[1]-b[1];
            }
        });
        List<int[]>  res=new LinkedList<>();        //保存结果
        for(int i=0;i<people.length;i++){
            int[] peo = people[i];
            res.add(peo[1], peo);
        }
        return res.toArray(new int[people.length][]);
    }


    //416. Partition Equal Subset Sum
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 == 1) return false;
        int target = sum/2;
        Arrays.sort(nums);
        return dfs(nums, 0, target);
    }

    boolean dfs(int[] nums, int pos, int target) {
        if (target == 0) return true;
        for (int i = pos; i < nums.length; ++i) {
            if (i > pos && nums[i] == nums[i-1]) continue;
            if (nums[i] > target) break;
            if (dfs(nums, i+1, target-nums[i])) return true;
        }
        return false;
    }

    //394. Decode String

    public String decodeString(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        Stack<Integer> number = new Stack<>();
        Stack<String> characters = new Stack<>();

        for (int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if (c >= '0' && c<= '9'){
                int j = i;
                int value =0;
                char tmp = s.charAt(j);
                while (tmp >= '0' && tmp <= '9'){
                    value = value * 10 + (tmp - '0');
                    j++;
                    tmp = s.charAt(j);
                }
                i = j - 1;
                number.push(value);
            }else if (c == '['){
                characters.push(stringBuilder.toString());
                stringBuilder.delete(0,stringBuilder.length());
            }else if(c == ']'){
                StringBuilder tmp = new StringBuilder(characters.pop());
                int value = number.pop();
               // String str = characters.pop();
                for (int x = 0; x < value; x++){
                    tmp.append(stringBuilder);
                }
                stringBuilder = tmp;
                System.out.println(tmp.toString());
            }else {
                stringBuilder.append(c);
            }
        }
        return stringBuilder.toString();

    }


    //437. Path Sum III
    //递归加 组合结果  首先我一个状态有左右子节点的解之和在加上根节点的递归得到
    //第二条思路   把数转换为数组进行求解  先使用先序遍历
    public int pathSum(TreeNode root, int sum) {
        if(root == null) return 0;
        return dfs(root,sum,0) + pathSum(root.left,sum) + pathSum(root.right,sum);

    }

    public int dfs(TreeNode root, int target, int sum){  //表示左右孩子中内有多少条路劲
        if(root == null) return 0;
        int cur = root.val + sum;
        int flag = cur == target? 1 : 0;
        return flag + dfs(root.left,target,cur) + dfs(root.right,target,cur);
    }



    //438. Find All Anagrams in a String  求出我们两个二进制数不同的位数  <<补上0  >>补上符号位  >>>补0
    /*
    * <<：左移运算符，num << 1,相当于num乘以2
    *  >>：右移运算符，num >> 1,相当于num除以2
    *  >>>：无符号右移，忽略符号位，空位都以0补齐
    * */
    public int hammingDistance(int x, int y) {
        int xor = x ^ y, count = 0;
        for (int i = 0; i < 32; i++) count += (xor >> i) & 1;
        return count;
    }



    //448. Find All Numbers Disappeared in an Array
    public List<Integer> findDisappearedNumbers(int[] nums) {
        LinkedList<Integer> result = new LinkedList<>();
        for (int i = 0; i < nums.length; i++){
            while(nums[i]!=i+1&&nums[nums[i]-1]!=nums[i])
            {
                swap(nums,i,nums[i]-1);
            }
        }
        for (int i = 0; i < nums.length; i++){
            System.out.print(nums[i]+" ");
            if (nums[i] != i+1)
                result.addLast(i+1);
        }
        return result;
    }

    //538. Convert BST to  Tree
    public TreeNode convertBST(TreeNode root) {
        int[] sum = {0};
        dfs(root,sum);
        return  root;
    }

    public void dfs(TreeNode root,int[] sum){
        if (root == null) return ;
        dfs(root.right,sum);
        root.val += sum[0];
        sum[0] = root.val;
        dfs(root.left,sum);
    }

    //494. Target Sum
    //方法一  递归
    public int findTargetSumWays(int[] nums, int S) {
        int[] ans = {0};

        dfsFindTargetSumWays(0,nums,0,S,ans);
        return ans[0];
    }
    public void dfsFindTargetSumWays(int index,int[] nums,long sum,int s,int[] ans){
        if (index >= nums.length){
            if (sum == s) ans[0]++;
            return;
        }
        dfsFindTargetSumWays(index+1,nums,sum+nums[index],s,ans);
        dfsFindTargetSumWays(index+1,nums,sum-nums[index],s,ans);
    }



    //438. Find All Anagrams in a String  哈希加滑动窗口
    public <slen> List<Integer> findAnagrams(String s, String p) {
        ArrayList<Integer> result = new ArrayList<>();
        if (s.length() == 0) return result;

        int[] hashMap = new int[128];
        char[] ss = s.toCharArray();
        char[] pp = p.toCharArray();
        for (char c : pp) hashMap[c]++;
        int i = 0;

        while (i < ss.length - pp.length + 1){
            boolean success = true;
            int[] map = new int[128];

            for (int j = 0;j < 128; j++) map[j] = hashMap[j];

            for (int j = i; j < i + pp.length ; ++j) {
                if (--map[ss[j]] < 0) {
                    success = false;
                    break;
                }
            }
            if (success)
                result.add(i);

            i++;

        }

        return result;
    }

    //543. Diameter of Binary Tree  计算出二叉树直径长度  即任意两个结点形成最大长度  是一道求树深度的变形
    public int diameterOfBinaryTree(TreeNode root) {
        int[] result = new int[1];
        result[0] = 0;
        deptDiameterOfBinaryTree(root,result);
        return result[0];

    }

    public int deptDiameterOfBinaryTree(TreeNode root,int[] result){
        if (root ==null) return 0;
        int left = deptDiameterOfBinaryTree(root.left,result);
        int right = deptDiameterOfBinaryTree(root.right,result);
        int sum = left +right ;
        result[0] = result[0] >sum ? result[0]:sum;
        return  left > right ? left+1:right+1;
    }


    //572. Subtree of Another Tree
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(s ==null && t ==null) return true;
        if(s == null)
            return false;

        boolean flag = dfsIsSubtree(s,t);
        if(flag) return true;
        else{
            return isSubtree(s.left,t) || isSubtree(s.right,t);
        }

    }

    public boolean dfsIsSubtree(TreeNode root, TreeNode t){
        if(root == null && t == null)  return true;
        if(root == null || t == null) return false;

        if(root.val == t.val)
            return dfsIsSubtree(root.left,t.left) && dfsIsSubtree(root.right,t.right);
        else return false;
    }

    //560. Subarray Sum Equals K
    /*
    * 方法一  我们先用一个dp【i】 数组 记录从0 到 i 的和  计算从j到k的和就使用dp【k】 - dp【i】
    *
    * 方法二 建立在方法一之上  我们使用hashmap 表示到达数字sum的方法有多少  当我能到达sum时，我们我去查阅到达sum-k的方法有多少
    *                           这样的话  我们就能得到能加和到k的方法有多少
    * */
    public int subarraySum(int[] nums, int k) {

        HashMap<Integer,Integer> hashMap = new HashMap<>();
        hashMap.put(0,1);
        int sum = 0;
        int result = 0;
        for(int num : nums){
            sum += num;
            int waysKey = sum - k;
            result +=  hashMap.get(waysKey) == null? 0 : hashMap.get(waysKey);
            if (!hashMap.containsKey(sum)) hashMap.put(sum,1);
            else  hashMap.replace(sum,hashMap.get(sum)+1);

        }

        return  result;
    }

    //581. Shortest Unsorted Continuous Subarray
    // 自己的思路  既然让我想出我们需要排序的数目，那么我们先把数组排序  首尾不符合的那一段区间就是所求
    //网上大部分思路都是如此
    public int findUnsortedSubarray(int[] nums) {
        int[] tmp = new int[nums.length];
        for (int i = 0; i < nums.length; i++)
            tmp[i] = nums[i];

        Arrays.sort(tmp);
        int ans = 0;
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            if (tmp[left] == nums[left]){
                ans++;
                left++;
            }else
                break;
        }

        while(left <= right){
            if (tmp[right] == nums[right]){
                ans++;
                right--;
            }
            else break;
        }
        return  nums.length - ans;
    }


    //617. Merge Two Binary Trees
    //第一感觉递归专题   我的思路  --》 =把两树进行加和  构建出一颗新的树
    //leetcode  速度最快的思路   把t2树加载t1之上
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null)
            return null;

        TreeNode root = new TreeNode(getTreeNodeValue(t1,t2));
        if (t1 == null){
            root.left = mergeTrees(null,t2.left);
            root.right = mergeTrees(null,t2.right);
        }else if (t2 == null){
            root.left = mergeTrees(t1.left,null);
            root.right = mergeTrees(t1.right,null);
        }
        else {
            root.left = mergeTrees(t1.left,t2.left);
            root.right = mergeTrees(t1.right,t2.right);
        }

        return  root;
        /*
        if(t1 == null)
            return t2;
        if(t2 == null)
            return t1;
        t1.val += t2.val;
        t1.left = mergeTrees(t1.left,t2.left);
        t1.right = mergeTrees(t1.right,t2.right);
        return t1;
        */
    }
    public int getTreeNodeValue(TreeNode t1, TreeNode t2){
        int v1 = 0;
        int v2 = 0;
        if (t1 != null) v1 = t1.val;
        if (t2 != null) v2 = t2.val;
        return v1+v2;
    }

    //771. Jewels and Stones
    //自己的思路  ---》 直接遍历 建立一个hashmap  直接求得   一半大家都能想到的思路
    public int numJewelsInStones(String J, String S) {
        int[] hashMap = new int[128];
        for (int i = 0; i < J.length(); i++){
            hashMap[J.charAt(i)]++;
        }

        int ans = 0;
        for (int i = 0; i < S.length(); i++){
            if (hashMap[S.charAt(i)]  > 0)
                ans++;
        }
        return ans;
    }

    //647. Palindromic Substrings
    //思路一  比较普通的思路 从中心向四周扩散的的算法  不过要分奇数偶数
    public int countSubstrings(String s) {

        int ans = 0;
        for (int i = 0; i < s.length(); i++){
            ans += count(s,i,i);
            ans += count(s,i,i+1);
        }
        return ans;

    }

    public int count(String s,int begin,int end){
        int ans = 0;
        while (begin >= 0 && end < s.length() && s.charAt(begin) == s.charAt(end)){
            ans++;

            begin--;
            end++;
        }
        return ans;
    }

    //621. Task Scheduler
    //把我们的频率最高的任务进行分块
    /*
    c[25]是出现最多的字母数，所以(c[25] - 1)是分块数,例如A（4）次数最多，间隔n（3）(A***A***A***A)则重复的段落数为4-1=3。间隔是n，包含领头的A在内，每个间隔的长度是n+1=3+1=4.
所以整个段落长度是(c[25] - 1) * (n + 1)。因为出现频率最高的元素可能不止一个，我们假设为k个，那么这种情况下最终的时间需求为：(c[25]-1)*（n+1）+k
    或者最终的长度为任务的数量。
    * */
    public int leastInterval(char[] tasks, int n) {
        int[] c = new int[26];
        for(char t : tasks){
            c[t - 'A']++;
        }
        Arrays.sort(c);
        int i = 25;
        while(i >= 0 && c[i] == c[25]) i--;

        return Math.max(tasks.length, (c[25] - 1) * (n + 1) + 25 - i);
    }
    //581、617,711 647   621

    //4 Median of Two Sorted Arrays
    //方法一  可以先归并成一个有序数组 然后直接获取中位数
    //方法二  类二分查找  下面代码即是
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length, left = (m + n + 1) / 2, right = (m + n + 2) / 2;
        return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0;
    }
    int findKth(int[] nums1, int i, int[] nums2, int j, int k) {
        if (i >= nums1.length) return nums2[j + k - 1];
        if (j >= nums2.length) return nums1[i + k - 1];
        if (k == 1) return Math.min(nums1[i], nums2[j]);
        int midVal1 = (i + k / 2 - 1 < nums1.length) ? nums1[i + k / 2 - 1] : Integer.MAX_VALUE;
        int midVal2 = (j + k / 2 - 1 < nums2.length) ? nums2[j + k / 2 - 1] : Integer.MAX_VALUE;
        if (midVal1 < midVal2)
            return findKth(nums1, i + k / 2, nums2, j, k - k / 2);

        return findKth(nums1, i, nums2, j + k / 2, k - k / 2);
    }

    //10. Regular Expression Matching
    /**
     * 一 、 这是一道关于有限自动状态机的题目   正则表达式  用p去匹配s  s使用的是回溯法
     *
     * 二 、 这道题还可以使用动态规划
     */
    public boolean isMatch(String s, String p) {
        return check(s.toCharArray(),0,p.toCharArray(),0);
    }
    public boolean check(char[] s,int i,char[] p,int j){
        if(i==s.length&&j==p.length) return true;
        if(i!=s.length&&j==p.length) return false;
        if(j+1<p.length){//第二个不为空
            if(i<s.length){//
                if(p[j+1]=='*'){
                    if(s[i]==p[j]||(p[j]=='.'&&i<s.length))
                        return check(s,i+1,p,j+2)||check(s,i+1,p,j)||check(s,i,p,j+2);
                    else
                        return check(s,i,p,j+2);
                }
            }
            else{//当i==s.length,j!=p.length 例如考虑s=a,p=ab*
                if(j+1<p.length&&p[j+1]=='*')
                    return check(s,i,p,j+2);
                else return false;
            }
        }
        if(i<s.length&&(s[i]==p[j]||p[j]=='.'))//因为最前面的if是判断j+1是否s.length，最后就判断最后一个字符
            return check(s,i+1,p,j+1);
        else
            return false;
    }
    //第二种思路： 使用动态规划

    private static boolean isMatch_dp(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];  // j 表示的 p   i表示的 s
        dp[0][0] = true;
        for (int i = 1; i < n; i++) { // 初始化第一行，p匹配s = ""
            if (p.charAt(i) == '*' && dp[0][i - 1]) {
                dp[0][i + 1] = true;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // p[j - 1]不是"*"的情况，单字符匹配
                if (p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                // p[j - 1]是"*"的情况，则要判断p[j - 2]是否匹配当前s[i - 1]
                // 若不匹配，则p[j - 1]匹配空字符串
                // 否则，有三种情况：
                //   1.p[j - 1]匹配空字符串；
                //   2.p[j - 1]匹配单一s[i - 1]字符；
                //   3.p[j - 1]匹配多个s[i - 1]字符
                if (p.charAt(j - 1) == '*') {
                    if (p.charAt(j - 2) != s.charAt(i - 1) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 2] || dp[i][j - 1] || dp[i - 1][j];
                        //分别对应a*   没有a   只有一个a          多个a
                    }
                }
            }
        }
        return dp[m][n];
    }

    //62. Unique Paths
    /**
     *  机器人过河
     *  好久没有   Runtime: 0 ms, faster than 100.00% of Java online submissions for Unique Paths.
     * Memory Usage: 31.7 MB, less than 100.00% of Java online submissions for Unique Paths.
     */

    public int uniquePaths(int m, int n) {
        if(m == 0 || n ==0)
            return 0;

        int dp[][] = new int[n][m];
        dp[0][0] = 1;
        for(int i = 1; i < m; i++)
            dp[0][i] = dp[0][i-1];
        for(int i = 1; i < n; i++)
            dp[i][0] = dp[i-1][0];

        for(int i = 1; i < n; i++){
            for(int j = 1; j < m; j++)
                dp[i][j] = dp[i-1][j]+dp[i][j-1];
        }

        return dp[n-1][m-1];
    }
    //11. Container With Most Water
    /**
     *  这道接雨水的题，如果先固定一个i坐标  j = i+ 1  j < heught.length() 这样求出两柱子之间的容量  维护一个最大值  时间复杂度O （n2）
     *  其实我们可以使用 双指针法  从首尾开始遍历  因为是要求最大容量  直接根据情况移动指针   时间复杂度 O（n）
     */
    //下面是使用双指针法
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int maxVol = 0;
        while (l < r) {
            int vol = Math.min(height[l], height[r]) * (r - l);
            if (vol > maxVol) {
                maxVol = vol;
            }
            if (height[l] > height[r]) {
                r --;
            } else {
                l ++;
            }
        }
        return maxVol;
    }


    //42. Trapping Rain Water
    /** 思路一：
     *   双指针法的思想  +  动态规划记录    常用思想  ：  对于一个数组每一个index 我们求出所需要index 左右 需要的值  进行运算
     *   这个思想在那一道  剑指offer之中的  取出除了当前index外的所有数的乘积
     *
     *思路二 ：
     *      对于两次遍历是多余的  我们还是只是  双指针法
     *      基本思路是这样的，用两个指针从两端往中间扫，在当前窗口下，如果哪一侧的高度是小的，那么从这里开始继续扫，
     *      如果比它还小的，肯定装水的瓶颈就是它了，可以把装水量加入结果，如果遇到比它大的，立即停止，重新判断左右窗口的大小情况，重复上面的步骤。
     *      这里能作为停下来判断的窗口，说明肯定比前面的大了，所以目前肯定装不了水（不然前面会直接扫过去）。
     *      这样当左右窗口相遇时，就可以结束了，因为每个元素的装水量都已经记录过了
     */
    public int trap(int[] height) {
        if ( height == null|| height.length == 0){
            return 0;
        }
        int dp[] = new int[height.length];
        int flagMax = 0;
        int result = 0;
        for (int i = 0; i < height.length; i++){
            dp[i] = flagMax;
            flagMax = Math.max(height[i],flagMax);
        }
        flagMax =  0;

        for (int i = height.length - 1; i >= 0; i--){
            dp[i] = Math.min(flagMax,dp[i]);  //我们为了找出  index  左右两个方向之中  高度最小的
            flagMax = Math.max(flagMax, height[i]);

            int t =  dp[i] - height[i];
            result += t > 0 ? t:0;
        }
        return result;
    }

    public int trap2(int[] height) {
        if ( height == null|| height.length == 0){
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int result = 0;
        while (left < right){

            int minF = Math.min(height[left],height[right]);

            if (height[minF] == height[left]){
                left++;
                while (left < right && height[left] < minF)
                    result += minF - height[left++];

            }else {
                right++;
                while (left < right && minF > height[right])
                    result += minF - height[right++];
            }
        }
        return result;
    }

    //128. Longest Consecutive Sequence
    /**
     *  求出数组中最长的连续子序列
     *
     *  可以使用hashMap 进行相邻数的遍历
     *
     *  使用hashSet （底层也是使用hashmap）的原理  每次取出的数要把其相邻数去掉
     *
     */
    public int longestConsecutive(int[] nums) {

        HashSet<Integer> hashSet = new HashSet<>();
        for (int num : nums)  hashSet.add(num);

        int res = 0;
        for (int num : nums){
            int pre = num - 1;
            int next = num + 1;

            while (hashSet.remove(pre)) pre--;
            while (hashSet.remove(next)) next++;
            res = Math.max(res,next - pre - 1);
        }
        return res;
    }

    public int longestConsecutive2(int[] nums) {
        int res = 0;
        Map<Integer, Integer> m = new HashMap<Integer, Integer>();
        for (int num : nums) {
            if (m.containsKey(num)) continue;
            int left = m.containsKey(num - 1) ? m.get(num - 1) : 0;
            int right = m.containsKey(num + 1) ? m.get(num + 1) : 0;
            int sum = left + right + 1;
            m.put(num, sum);
            res = Math.max(res, sum);
            m.put(num - left, sum);
            m.put(num + right, sum);
        }
        return res;
    }

    //239. Sliding Window Maximum
    /**
     * 双端队列  维护一个排序的窗口   窗口第一个数字即为所求
     *  每个数据用了两次  时间复杂度是数组长度的两倍
     * 使用
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0)
            return new int[0];

        int res[] = new int[nums.length - k +1];

        LinkedList<Integer> deque = new LinkedList<>();

        //int index = 0;
        for (int i = 0; i < nums.length; i++){
            if (!deque.isEmpty() && deque.peekFirst() == i - k) deque.poll();

            while (!deque.isEmpty() && nums[i] > nums[deque.getLast()]) deque.removeLast();

            deque.add(i);

            if((i + 1) >= k) res[i + 1 - k] = nums[deque.peek()];

        }
        return res;
    }

    //23. Merge k Sorted Lists

    /*
    堆排序
    分治
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0)
            return null;
        if(lists.length == 1)
            return lists[0];
        return recursion(lists,0,lists.length - 1);
    }
    //recursion
    public ListNode recursion(ListNode[] lists,int start,int end){
        if(start == end)//只有一个链表
            return lists[start];
        if(start < end){
            int mid = start + (end - start) / 2; //注意：这里防止整数越界的处理,start+(end-start)/2
            ListNode l1 = recursion(lists,start,mid);
            ListNode l2 = recursion(lists,mid + 1,end);
            return merge2(l1,l2);
        } else
            return null;

    }
    //merge
    public ListNode merge2(ListNode l1,ListNode l2){
        ListNode head = new ListNode(0); //创建一个头结点，最后还要删掉
        ListNode p = head;
        while(l1 != null && l2 != null){
            if(l1.val <= l2.val){
                p.next = l1;
                l1 = l1.next;
            } else{
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }

        p.next = (l1 != null) ? l1 : l2;
        return head.next;// head的下一个节点是第一个数据结点
    }

    //72. Edit Distance

    /**
     *
     * @param word1
     * @param word2
     * @return 两个单词的距离
     *
     * 假设自由删除插入
     * 接下来，定义一个表达式D(i,j)。它表示从第1个字单词的第0位至第i位形成的子串和第2个单词的第0位至第j位形成的子串的编辑距离。
     *
     * 显然，可以计算出动态规划的初始表达式，如下:
     *
     * D(i,0) = i
     *
     * D(0,j) = j
     *
     * 然后，考虑动态规划的状态转移方程式，如下:
     *
     *                                    D(i-1, j) + 1
     * D(i,j)=min                                 D(i, j-1) + 1 
     *                                    D(i-1, j-1) +2    ( if  X(i) != Y(j) ) ; leetode的替换相当于删除插入
     *                                                      D(i-1,j-1)     ( if  X(i) == Y(j) )

     */
    public int minDistance(String word1, String word2) {
        if (word1.length()==0)
            return word2.length();
        //
        if (word2.length() ==0)
            return word1.length();
        int[][] dp = new int[word1.length()+1][word2.length()+1];

        for (int i = 1; i<= word1.length(); i++)
            dp[i][0] = i;

        for (int i = 1; i<= word2.length(); i++)
            dp[0][i] = i;

        for (int i = 1; i <= word1.length(); i++){
            for (int j = 1; j <= word2.length(); j++){
                int replace_step = 0;
                if (word1.charAt(i-1) == word2.charAt(j-1))
                    replace_step = dp[i-1][j-1];
                else
                    replace_step =  dp[i - 1][j - 1] + 1;

                replace_step = Math.min(dp[i-1][j]+1,replace_step);
                dp[i][j] = Math.min(replace_step,dp[i][j-1]+1);
            }
        }
        return dp[word1.length()][word2.length()];
    }


    //84. Maximal Rectangle
    /*题目
    *  Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.
    *
    * 1、如果已知height数组是升序的，应该怎么做？

    比如1,2,5,7,8

    那么就是(1*5) vs. (2*4) vs. (5*3) vs. (7*2) vs. (8*1)

    也就是max(height[i]*(size-i))

    2、使用栈的目的就是构造这样的升序序列，按照以上方法求解。

    但是height本身不一定是升序的，应该怎样构建栈？

    比如2,1,5,6,2,3

    （1）2进栈。s={2}, result = 0

    （2）1比2小，不满足升序条件，因此将2弹出，并记录当前结果为2*1=2。

    将2替换为1重新进栈。s={1,1}, result = 2

    （3）5比1大，满足升序条件，进栈。s={1,1,5},result = 2

    （4）6比5大，满足升序条件，进栈。s={1,1,5,6},result = 2

    （5）2比6小，不满足升序条件，因此将6弹出，并记录当前结果为6*1=6。s={1,1,5},result = 6

    2比5小，不满足升序条件，因此将5弹出，并记录当前结果为5*2=10（因为已经弹出的5,6是升序的）。s={1,1},result = 10

    2比1大，将弹出的5,6替换为2重新进栈。s={1,1,2,2,2},result = 10

    （6）3比2大，满足升序条件，进栈。s={1,1,2,2,2,3},result = 10

    栈构建完成，满足升序条件，因此按照升序处理办法得到上述的max(height[i]*(size-i))=max{3*1, 2*2, 2*3, 2*4, 1*5, 1*6}=8<10

    综上所述，result=10
     */
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> sortStack = new Stack<>();

        int result = 0;
        for (int i = 0; i < heights.length; i++){
            if (sortStack.isEmpty() || sortStack.peek() <= heights[i])
                sortStack.push(heights[i]);
            else
            {

                int count = 1;
                while(!sortStack.isEmpty() && sortStack.peek() > heights[i])
                {

                    result = Math.max(result, sortStack.peek()*count);
                    sortStack.pop();
                    count++;
                }
                while(count-- != 0)
                    sortStack.push(heights[i]);
                sortStack.push(heights[i]);

            }
        }
        int count = 1;
        while(!sortStack.isEmpty())
        {
            result = Math.max(result, sortStack.peek()*count);
            sortStack.pop();
            count ++;
        }
        return result;
    }

    //85. Maximal Rectangle

    /**
     * 思路一  根据84的思路  把二维合成一维  在每一行（与这一行往上的和）求他和的加和  使用84的算法
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0] == null) return 0;

        int m = matrix.length, n = matrix[0].length;
        int[] l = new int[n];
        int[] r = new int[n];
        int[] h = new int[n];
        int result = 0;

        for(int i = 0; i < n; i++){
            l[i] = 0;
            r[i] = n;
            h[i] = 0;
        }
        for(int i = 0; i < m; i++){
            int cur_left = 0, cur_right = n;
            for(int j = 0; j < n; j++){
                if(matrix[i][j] == '1') h[j] += 1;
                else                    h[j] = 0;
//              System.out.print(h[j]);
//              System.out.print(" ");
            }
            for(int j = 0; j < n; j++){
                if(matrix[i][j] == '1'){
                    l[j] = Math.max(l[j], cur_left);
                }
                else{
                    l[j] = 0;
                    cur_left = j + 1;
                }
//              System.out.print(l[j]);
//              System.out.print(" ");
            }
            for(int j = n-1; j >= 0; j--){
                if(matrix[i][j] == '1'){
                    r[j] = Math.min(r[j], cur_right);
                }
                else{
                    r[j] = n;
                    cur_right = j;
                }
//              System.out.print(r[j]);
//              System.out.print(" ");
            }
            for(int j = 0; j < n; j++){
                result = Math.max(result, (r[j] - l[j]) * h[j]);
            }
            System.out.println();
        }

        return result;

    }







}