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




}