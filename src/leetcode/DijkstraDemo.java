package leetcode;


import org.junit.Test;

public class DijkstraDemo {
    //表示两个顶点之间不可直达的值
    public static final int I = 0xFFFF;

    static int[][] array = new int[][]{
            {0, 1, 5, I, I, I, I, I, I},
            {1, 0, 3, 7, 5, I, I, I, I},
            {5, 3, 0, I, 1, 7, I, I, I},
            {I, 7, I, 0, 2, I, 3, I, I},
            {I, 5, 1, 2, 0, 3, 6, 9, I},
            {I, I, 7, I, 3, 0, I, 5, I},
            {I, I, I, 3, 6, I, 0, 2, 7},
            {I, I, I, I, 9, 5, 2, 0, 4},
            {I, I, I, I, I, I, 7, 4, 0}
    };

    //最短路径数组, 对应的前驱索引值
    int[] path = new int[array.length];

    //最短权重数组, 存放每次到另个顶点的修改后的权重值，先修改第一行V0
    int[] weight = array[0];

    /**
     * 单源最短路径算法
     */
    public void dijkstra() {
        int k = 0;//表示当前正要处理的顶点Vk

        //定义一个数组来存放集合U 和集合V 两个集合的顶点的信息
        boolean[] flag = new boolean[array.length];
        //先从第一个顶点开始，所以先直接存放在U集合一边
        flag[0] = true;
        //开始逻辑,求VO到某个顶点的最短路径
        for (int v = 1; v < array.length; v++) {
            //在能走的路径中找到最短的一条，也就是在集合V 中找
            int min = I;
            for (int i = 0; i < array.length; i++) {
                if (!flag[i] && weight[i] < min) {
                    k = i;//K为U集合到V集合中找到的顶点
                    min = weight[i];//min找到了最小值的位置
                }
            }
            //找到最短路径后，把它存放在集合U中，然后从这个最短的路径对应的顶点开始找下一轮
            flag[k] = true;
            for (int i = 0; i < array.length; i++) {
                if (!flag[i] && min + array[k][i] < weight[i]) {
                    weight[i] = min + array[k][i];//修改路径长度
                    path[i] = k;//保存前驱
                }
            }
        }
    }

    @Test
    public void main() {
        dijkstra();

        //打印
        System.out.print("最短路径：");
        for (int i = 0; i < path.length; i++) {
            System.out.print(path[i] + " ");
        }
        System.out.println();
        System.out.print("最短权重：");
        for (int i = 0; i < weight.length; i++) {
            System.out.print(weight[i] + " ");
        }

        System.out.println();
        System.out.print("V8节点的最短路径：");
        //打印结果
        int v = 8;
        while (v != 0) {
            System.out.print(path[v]);
            v = path[v];
        }
    }

}
