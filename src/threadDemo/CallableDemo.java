package threadDemo;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

public class CallableDemo implements Callable<Integer> {
    //实现一个数组的求和

    int start;
    int end;

    public CallableDemo (int start, int end){
        this.start = start;
        this.end = end;

    }

    private static volatile  int[] nums;
    {
        nums = new int[1000];
        for (int i = 0; i < 1000; i++)
            nums[i] = i;
    }


    @Override
    public Integer call() throws Exception {
        int tmpResult = 0;
        for (int i = start; i < end; i++){
            tmpResult += nums[i];
          //  System.out.println(Thread.currentThread().getName() + " 正在工作 " );
        }
        return tmpResult;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        FutureTask<Integer>[] result = new FutureTask[4];
        for (int i = 0; i < 4; i++){
            result[i] = new FutureTask<>(new CallableDemo(i*250,(i+1)*250));
            new Thread(result[i]).start();

        }

        int res = 0;
        for (int i = 0;i < 4; i++)
            res += result[i].get();

        System.out.println("result " + res);

    }


}
