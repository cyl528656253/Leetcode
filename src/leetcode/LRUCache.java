package leetcode;


import java.util.HashMap;

//146 146. LRU Cache

/**
 * 这是一道使用双向链表加上一个hashmap的题目  使用head和end两个两个指针记录头尾
 * 使用map来控制O(1)的时间取值   链表的删除增加是O（1）
 * 下面设计是没有头结点的设计  其实在头尾设计头结点尾节点速度回更更快
 *
 */
public class LRUCache {
    int capacity;
    HashMap<Integer, LRUNode> map = new HashMap<Integer, LRUNode>();
    LRUNode head=null;//先声明一个头结点和一个尾节点
    LRUNode end=null;


    public LRUCache(int capacity) {
        this.capacity = capacity;
    }


    public int get(int key) {
        if (map.containsKey(key)){
            LRUNode node = map.get(key);
            remove(node);
            toHead(node);
            return node.value;
        }

        return -1;
    }


    public  void  remove(LRUNode node){
        if (node.pre != null){
            node.pre.next = node.next;
        }else {
            head = node.next;
        }

        if (node.next != null){
            node.next.pre = node.pre;
        }else {
            end = node.pre;
        }

    }

    public void toHead(LRUNode node){
        node.next = head;
        node.pre = null;

        if (head!= null)
            head.pre = node;

        head = node;

        if (end==null)
            end = head;
    }

    public void put(int key, int value) {

        if (map.containsKey(key)){
            LRUNode node = map.get(key);
            node.value = value;
            remove(node);
            toHead(node);
        }else {
            LRUNode node = new LRUNode(key,value);
            if (map.size() >= capacity){
                map.remove(end.key);
                remove(end);
            }
            toHead(node);
            map.put(key,node);
        }
    }

}
/**
 *  有首尾结点的设计
 *  class LRUCache {
 *     private class Node{
 *         int key;
 *         int value;
 *         Node prev;
 *         Node next;
 *         public Node(int key, int value){
 *             this.key = key;
 *             this.value = value;
 *             prev = null;
 *             next = null;
 *         }
 *     }
 *     private int capacity;
 *     private HashMap<Integer, Node> map;
 *     private Node head;
 *     private Node tail;
 *
 *     public LRUCache(int capacity) {
 *         this.capacity = capacity;
 *         map = new HashMap<Integer, Node>();
 *         head = new Node(-1,-1);
 *         tail = new Node(-1, -1);
 *         head.next = tail;
 *         tail.prev = head;
 *     }
 *
 *     public int get(int key) {
 *         if(!map.containsKey(key)) return -1;
 *
 *         Node currNode = map.get(key);
 *         remove(currNode);
 *
 *         return add(currNode);
 *     }
 *
 *     public void put(int key, int value) {
 *         if(get(key) != -1){
 *             map.get(key).value = value;
 *             return;
 *         }
 *         if(map.size() == capacity){
 *             map.remove(head.next.key);
 *             remove(head.next);
 *         }
 *
 *         Node node = new Node(key, value);
 *         add(node);
 *         map.put(key, node);
 *     }
 *
 *     private void remove(Node node){
 *         node.prev.next = node.next;
 *         node.next.prev = node.prev;
 *     }
 *
 *     private int add(Node node){
 *         node.prev = tail.prev;
 *         node.next = tail;
 *         node.prev.next = node;
 *         node.next.prev = node;
 *         return node.value;
 *     }
 * }
 */

