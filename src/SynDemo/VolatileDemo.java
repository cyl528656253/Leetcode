package SynDemo;

//volatile 用于代码的顺序统一   对于一些语句如 n++ ： read n   inc c   write n  可能会出现脏读


public class VolatileDemo {

    public static volatile int race = 0;
    public static void increase() {
        race++;

    }
    public static int THREADS_COUNT = 3;

    public static void main(String[] args) throws InterruptedException {
        Thread[] threads = new Thread[THREADS_COUNT];
        for (int i = 0; i < THREADS_COUNT; i++) {
            threads[i] = new Thread(new Runnable() {
                public void run() {
                    for (int i = 0; i < 100000; i++) {
                        increase();

                    }
                }
            });

            threads[i].start();
        }

        while (Thread.activeCount() > 1) {
            //Thread.yield();
            Thread.sleep(150);
            System.out.println(race);
        }
    }
}
