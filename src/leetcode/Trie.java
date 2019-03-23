package leetcode;



//前缀树  或者叫做  字典树
public class Trie {
    private TrieNode root;



    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode cur = root;
        for (int i = 0; i < word.length(); i++){
            int index = word.charAt(i) - 'a';
            if (cur.child[index] == null) cur.child[index] = new TrieNode();
            cur = cur.child[index];
        }
        cur.word = true;
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode cur = root;
        for (int i = 0; i < word.length(); i++){
            int index = word.charAt(i) - 'a';
            if (cur.child[index] == null) return false;
            cur = cur.child[index];
        }
        return cur.word;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode cur = root;
        for (int i = 0; i < prefix.length();i++){
            int index = prefix.charAt(i) - 'a';
            if (cur.child[index] == null) return false;
            cur = cur.child[index];
        }
        return true;
    }



}
