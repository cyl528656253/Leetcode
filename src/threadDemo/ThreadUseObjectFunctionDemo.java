package threadDemo;


import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

public class ThreadUseObjectFunctionDemo implements Runnable {


    static int value = 0;

    @Override
    public void run() {
        while (value <= 100) {
            synchronized (ThreadUseObjectFunctionDemo.class) {
                System.out.println(Thread.currentThread().getName() + ":" + value++);
                ThreadUseObjectFunctionDemo.class.notify();
                try {
                    ThreadUseObjectFunctionDemo.class.wait();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void main(String[] args) {
        new Thread(new ThreadUseObjectFunctionDemo(), "偶数").start();
        new Thread(new ThreadUseObjectFunctionDemo(), "奇数").start();
    }
}
/*
    public static void main(String[] args) {
        int worker = 3;
        AtomicInteger countDownLatch = new AtomicInteger(worker);
        new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("D is waiting for other three threads");
                try {
                    countDownLatch.await();
                    System.out.println("All done, D starts working");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
        for (char threadName='A'; threadName <= 'C'; threadName++) {
            final String tN = String.valueOf(threadName);
            new Thread(new Runnable() {
                @Override
                public void run() {
                    System.out.println(tN + "is working");
                    try {
                        Thread.sleep(100);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    System.out.println(tN + "finished");
                    countDownLatch.countDown();
                }
            }).start();
        }
    }

*/

