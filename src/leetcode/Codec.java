package leetcode;


//297. Serialize and Deserialize Binary Tree

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

/**
 *  层次遍历  取出序列化
 *  序列化后的重新建树
 *
 *  细节很难   比如逗号  中括号  等等
 */

public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder arr = new StringBuilder();
        arr.append("[");
        if(root == null) return "[]";
        ArrayList<TreeNode> queue = new ArrayList<>();
        queue.add(root);
        arr.append(root.val);
        for (int i = 0; i < queue.size(); i++) {
            TreeNode node = queue.get(i);
            if(node == null) continue;
            queue.add(node.left);
            queue.add(node.right);
        }

        while(queue.get(queue.size()-1) == null){   //消去leaf的null
            queue.remove(queue.size()-1);
        }

        for(int i = 1 ; i < queue.size(); i++){
            TreeNode node = queue.get(i);
            if(node == null)
                arr.append(",").append("null");
            else
                arr.append(",").append(node.val);

        }
        arr.append("]");
        System.out.println(arr.toString());
        return arr.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.length() == 0)
            return null;

        if(data == null || data.length() == 0 || data.equals("[]")) return null;
        String[] arr = data.substring(1, data.length() - 1).split(",");
        if(arr.length == 0) return null;
        ArrayList<TreeNode> list = new ArrayList<TreeNode>();
        TreeNode root = new TreeNode(Integer.parseInt(arr[0]));
        list.add(root);
        int index = 0;
        boolean isLeftNode = true;
        for(int i = 1 ; i < arr.length ; i++){
            if(!arr[i].equals("null")){
                TreeNode node = new TreeNode(Integer.parseInt(arr[i]));
                if(isLeftNode)
                    list.get(index).left = node;
                else
                    list.get(index).right = node;
                list.add(node);
            }
            if(!isLeftNode)  index++;
            isLeftNode = !isLeftNode;
        }
        return root;
    }

    /*
    public String serialize(TreeNode root) {
        if (root == null) {
            return "null";
        }
        return "" + root.val + "," +serialize(root.left) + "," + serialize(root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] vals = data.split(",");
        int[] index = new int[]{0};
        TreeNode result = helper(vals, index);
        return result;
    }

    private TreeNode helper(String[] vals, int[] index) {
        if (vals[index[0]].equals("null")) {
            index[0]++;
            return null;
        }
        TreeNode node = new TreeNode(Integer.valueOf(vals[index[0]]));
        index[0]++;
        node.left = helper(vals, index);
        node.right = helper(vals, index);
        return node;
    }
    */


}
