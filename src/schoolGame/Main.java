package schoolGame;

import java.util.Scanner;
//线上笔试题  A数组和B数组交换一个数让数组综合相同
public class Main {

    public  static int pow(int x, int y, int mod) {
        int res = 1;
        while(y != 0)
        {
            if( (y & 1) == 1) {
                res =  (res * x) % mod;
            }
            x = (x * x) % mod;
            y = y / 2;
        }
        return res;
    }

    public  static  void main(String[] args){
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        for (int zz = 0; zz < n; zz++){
            int nanNumber = scanner.nextInt();
            int nvNumber = scanner.nextInt();
            int D = scanner.nextInt();

            int male[] = new int[D+3];
            int female[] = new int[nvNumber];

            for (int i = 0; i < nanNumber; i++){
                int k = scanner.nextInt();
                male[pow(k,k,D)]++;
              //  System.out.println(" i " + i + "  ans "+pow(k,k,D));
            }

            int index1 = 0;
            for (int i = 0; i < nvNumber; i++){
                int k = scanner.nextInt();
                female[index1++] = k;
            }
          //  System.out.println( " D  " + D);
            int ans = 0;
            for (int i = 0; i < female.length; i++){
                int index = female[i];
               // System.out.println("female " + index);
                while (index < D)   {
                    if (male[index] > 0){
                    //    System.out.println(index + "  ss ");
                        male[index] =  male[index]  -1 ;
                        ans++;
                        break;
                    }else {
                        index++;
                       // System.out.println("index  --> "+index);
                    }
                }

            }
            System.out.println(ans);

        }





        /*
        Scanner scanner = new Scanner(System.in);
        String s = scanner.nextLine();
        int a1 = 0;
        int a0 = 0;
        for (char c : s.toCharArray()){
            if (c == '1')
                a1++;
            else
                a0++;
        }
        int min = a1>a0? a0:a1;
        System.out.println(s.length() - min * 2 );
        */




        /*
        String ss = "Once upon a time, there was a mountain. Over the mountain there was a temple. In the temple, there was an old monk telling a story. This is the story: ";

        StringBuilder sb = new StringBuilder();
        for (char c : ss.toCharArray()){
            if (Character.isLetter(c))
                sb.append(c);
        }
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
       // n--;
        n = n % sb.length();

        if (n == 0)
            n = sb.length();

       char c = sb.charAt(n-1);

       System.out.println(Character.toLowerCase(c));
*/
    }
}
