package test;


import leetcode.TreeNode;
import org.junit.Test;

import java.sql.Array;
import java.util.*;

public class SwordReferOffer {



    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {


        if (pre == null || in == null) {
            return null;
        }
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        if (pre.length != in.length) {
            return null;
        }
        TreeNode root = new TreeNode(pre[0]);//第一个
        for (int i = 0; i < in.length; i++) {
            if (pre[0] == in[i]) {
                //pre的0往后数i个是左子树的，copyofrange包含前面的下标，不包含后面的下标
                //in的i往前数i个是左子树的。
                root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                //注意in是从i+1开始，因为i是现在的根，i+1开始才是右子树
                root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length),
                        Arrays.copyOfRange(in, i + 1, in.length));
            }
        }
        return root;
    }


        public TreeNode reConstructBinaryTree2(int [] pre, int [] in) {
        if(pre == null || in == null || pre.length <= 0 || in.length <= 0)
            return null;

        return build(pre,in,0,pre.length-1,0,in.length-1);

    }

    //两个闭合区间
    public TreeNode build(int[] pre,int[] in,int startpre,int endpre, int startin,int endin) {
        int rootValue = pre[startpre];
        TreeNode treeNode = new TreeNode();
        treeNode.val = rootValue;
        int index = startin;

        if (startpre == endpre){
            return treeNode;
        }else if (startpre > endpre)
        {
            System.out.println("wrong");
            return null;
        }


        while (index <= endin && in[index] != rootValue){
            index++;
        }

        int leftBeginIn = startin;
        int leftEndIn = index - 1;
        int leftBeginPre = startpre + 1;
        int leftEndPre = leftBeginPre + (leftEndIn - leftBeginIn);

        int rightBeginIn = index + 1;
        int rightEndIn = endin;
        int rightBeginPre = leftEndPre + 1;
        int rightEndPre = endpre;

        if (index > startin){
            treeNode.left = build(in,pre,leftBeginPre,leftEndPre,leftBeginPre,leftEndPre);
        }
        if (index < endin)
            treeNode.right = build(in,pre,rightBeginPre,rightEndPre,rightBeginIn,rightEndIn);


        return treeNode;


    }



    public void backtrack(int n,
                          ArrayList<Integer> nums,
                          List<List<Integer>> output,
                          int first) {
        // if all integers are used up
        if (first == n)
            output.add(new ArrayList<Integer>(nums));
        for (int i = first; i < n; i++) {
            // place i-th integer first
            // in the current permutation
            Collections.swap(nums, first, i);
            // use next integers to complete the permutations
            backtrack(n, nums, output, first + 1);
            // backtrack
            Collections.swap(nums, first, i);
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        // init output list
        List<List<Integer>> output = new LinkedList();

        // convert nums into list since the output is a list of lists
        ArrayList<Integer> nums_lst = new ArrayList<Integer>();

        for (int num : nums)
            nums_lst.add(num);

        int n = nums.length;
        backtrack(n, nums_lst, output, 0);
        return output;
    }

    /**
     * 解题思路：
     * 整体思路类似，参考46-全排列题解 {@link https://leetcode-cn.com/problems/permutations/solution/46-quan-pai-lie-java-hui-su-suan-fa-by-pphdsny/}，其中需要注意的一点是，重复数字再取的时候得跳过
     * 1.先对数组进行排序
     * 2.采用回溯算法进行取数，当即将取到的数和之前回退回来的数一致的时候，再向上一层回溯
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();

        List<Integer> numsList = toList(nums);
        Collections.sort(numsList);
        List<Integer> item = new ArrayList<>();

        dfs(numsList,res,item,nums.length);
        return res;
    }

    //递归回溯
    private void dfs(List<Integer> numList,
                     List<List<Integer>> retList,
                     List<Integer> itemList,
                     int n) {

        if (itemList.size() == n)
        {
            retList.add(new ArrayList<>(itemList));
            return;
        }
        Integer pre = null;

        String s = "223";


        for (int i = 0; i < numList.size(); i++){

            if (pre != null && numList.get(i).equals(pre)){

                continue;
            }

            Integer value = numList.remove(i);
            itemList.add(value);
            dfs(numList,retList,itemList,n);
            itemList.remove(itemList.size() - 1);
            numList.add(i,value);
            pre = value;
        }
    }

    private List<Integer> toList(int[] nums) {
        List<Integer> retList = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            retList.add(nums[i]);
        }
        return retList;
    }


/*
    @Test
    public void findRepeatedNumber(){
        int[] arr1 = {5,7,32,656,5,1,4};
        int[] arr2= {3,7,9,45,77,5,100,5};

        Arrays.sort(arr1);
        Arrays.sort(arr2);
        Collections.sort(2, new Comparator<T>() {
            @Override
            public int compare(T o1, T o2) {
                    return 0;
            }
        });
    }
*/

    /**
     * 无重复字符的最长子串
     */

    public int lengthOfLongestSubstring(String s) {
        int[] map = new int[128];
        Arrays.fill(map, -1);
        int result = 0;
        int left = -1;

        for (int i = 0 ; i < s.length(); i++){
            char c = s.charAt(i);
            left = Math.max(map[c],left);
            map[c] = i;
            result = Math.max(result,i - left);
        }

        return result;


    }



    @Test
    public void arr(){

        ArrayList<Integer> arrayList = new ArrayList<>(4);
        for (int i = 0; i < 10; i++){
            arrayList.add(i);
            System.out.println(i + ":  list " + arrayList.size());
        }

    }






}
