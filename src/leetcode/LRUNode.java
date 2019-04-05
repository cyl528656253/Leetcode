package leetcode;

public class LRUNode {

    int key;
    int value;

    LRUNode pre;
    LRUNode next;


    LRUNode(int key,int value){
        this.key = key;
        this.value = value;
    }

}
