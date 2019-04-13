package threadPool;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

public class CallableDemo implements Callable<ArrayList<Integer>>{

    private int start;
    private int end;
    private int coreNumber;



    public void  setCoreNumber(int coreNumber){
        this.coreNumber = coreNumber;
    }




    public CallableDemo(int start,int end){
        this.start = start;
        this.end = end;
    }

    public CallableDemo(){

    }

    public boolean isSushu(int a){

        for (int i = 2; i <= (int)Math.sqrt(a); i++ ){
            if (a % i == 0)
                return false;
        }
        return true;
    }


    @Override
    public ArrayList<Integer> call() throws Exception {
        ArrayList<Integer> tmp = new ArrayList<>();
        for (int i = start;i <= end; i++){
            if (isSushu(i))
                tmp.add(i);
        }

        return tmp;
    }

    public static  void main(String[] args) throws Exception {
        CallableDemo demo = new CallableDemo(1,100);
        FutureTask<ArrayList<Integer>> result = new FutureTask<ArrayList<Integer>>(demo);
        new Thread(result).start();
        ArrayList<Integer> arrayList = result.get();
         for (int i = 0; i < arrayList.size(); i++)
             System.out.println(arrayList.get(i));

         return;

    }
}
