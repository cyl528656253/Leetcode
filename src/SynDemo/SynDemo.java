package SynDemo;

public class SynDemo implements Runnable{

//synchronized  可以修饰代码片

    @Override
    public void run() {
        synchronized (this){
            for (int i = 0;i < 10 ; i++){
                System.out.println(Thread.currentThread().getName() + "   : " + i);
            }
        }
    }


    public static void main(String[] args){
        SynDemo synDemo = new SynDemo();
        synDemo.run();
        new SynDemo().run();;
        new SynDemo().run();
    }
}
