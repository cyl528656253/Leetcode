package threadPool;


import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.concurrent.*;

//多线程找求数组和
public class ThreadPoolDemo{
    private long[] nums;
    private int coreNumber = 4;
    public long  result = 0;
    public Future<Long>[] futures = new Future[coreNumber];


    public void setNums(long[] nums){
        this.nums = nums;
        cachedThreadPool.prestartAllCoreThreads();

    }
    // newCachedTreadPool 建一个可缓存线程池，如果线程池长度超过处理需要，可灵活回收空闲线程，若无可回收，则新建线程
    private  ThreadPoolExecutor cachedThreadPool = new ThreadPoolExecutor(4,10,
            1000 * 60 ,TimeUnit.DAYS,new LinkedBlockingDeque<>());




    public static  void main(String[] args) throws Exception {
        long[] nums = new long[12345];
        for (int i = 0; i < 12345; i++)
            nums[i] = i;

        ThreadPoolDemo threadPoolDemo = new ThreadPoolDemo();
        threadPoolDemo.setNums(nums);
        long startTime=System.nanoTime();
        threadPoolDemo.doIt();
        long endTime=System.nanoTime(); //获取结束时间
    //    System.out.println("result " + threadPoolDemo.result);
        System.out.println("程序运行时间： "+(endTime-startTime)+"ns");


        long res = 0;
         startTime=System.nanoTime();
         for (int i = 0; i < 12345 ; i++)
            res += i;

        endTime=System.nanoTime(); //获取结束时间
        System.out.println("程序运行时间：--------------------- "+(endTime-startTime)+"ns");
    //    System.out.println("result " + res);

        threadPoolDemo.result = 0;
        startTime=System.nanoTime();
        threadPoolDemo.doIt();
        endTime=System.nanoTime(); //获取结束时间
     //   System.out.println("result " + threadPoolDemo.result);
        System.out.println("程序运行时间： "+(endTime-startTime)+"ns");

        return;
    }



    public void doIt() throws ExecutionException, InterruptedException {
        try {

            for (int i = 0; i < coreNumber; i++) {
                int start = i * nums.length / coreNumber + 1;
                int end1 = (i + 1) * nums.length / coreNumber;
                if (end1 > nums.length - 1)
                    end1 = nums.length - 1;
                int finalEnd = end1;
             //   System.out.println("start " + start + "  end " + finalEnd);
                Future<Integer> resultFuture;
                        Callable<Long> runnable = new Callable() {
                    @Override
                    public Long call() {
                        Long tmpResult = Long.valueOf(0);
                        for (int j = start; j <= finalEnd; j++) {
                           // System.out.println(j);
                                tmpResult += nums[j];
                        }
                        return tmpResult;
                    }
                };

                futures[i] = cachedThreadPool.submit(runnable);
            }
        }finally {

            for (int i = 0; i < futures.length;i++)
                result += futures[i].get();

        }



    }
}
