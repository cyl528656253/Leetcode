package SynDemo;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ReenTrantLockDemo  implements  Runnable{

     private  Lock lock = new ReentrantLock();  //默认非公平锁

    @Override
    public void run() {

        lock.lock();
        try {
            for (int  i = 0;i < 10; i++){
                System.out.println(Thread.currentThread().getName() + "  : " + i);
            }
        }finally {
            lock.unlock();
        }
    }


    public static void main(String[] args){
        new ReenTrantLockDemo().run();
        new ReenTrantLockDemo().run();
    }


    //这两个锁都是课可重入锁
}
