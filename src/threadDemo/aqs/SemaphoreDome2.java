package threadDemo.aqs;

import java.util.concurrent.Semaphore;

public class SemaphoreDome2 {
    static int result = 0;
    public static void main(String[] args) throws InterruptedException {
        int N = 3;

        Thread[] threads = new Thread[N];

        final Semaphore[] syncObjects = new Semaphore[N];
        for (int i = 0; i < N; i++) {
            syncObjects[i] = new Semaphore(1);
            if (i != N-1){
                syncObjects[i].acquire();
            }
        }
        for (int i = 0; i < N; i++) {
            final Semaphore lastSemphore = i == 0 ? syncObjects[N - 1] : syncObjects[i - 1];
            //上一个线程 有下一个线程的信号
            final Semaphore curSemphore = syncObjects[i];
            final int index = i;
            threads[i] = new Thread(new Runnable() {
                public void run() {
                    try {
                        while (true) { //先让上一个线程请求执行  这个线程在释放许可  就可以控制
                            lastSemphore.acquire();   //acquire 是一个堵塞的方法
                            System.out.println("thread" + index + ": " + result++);
                            if (result > 100){
                                System.exit(0);
                            }
                            curSemphore.release();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                }
            });
            threads[i].start();
        }
    }

}
