package io;

import java.io.*;

public class ReadAndWrite {

    public static  void main(String[] args){
        ReadAndWrite.test();


    }

    public static void test(){
        File firstFile = new File("D://io.txt");
        File secondFile=new File("D://io2.txt");
        BufferedReader in = null;
        BufferedWriter out = null;
        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(firstFile), "utf-8"));
            out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(secondFile), "utf-8"));
            String line = "";
            while((line = in.readLine())!=null){
                System.out.println(line);
                out.write(line+"\r\n");
            }
        } catch (FileNotFoundException e) {
            System.out.println("file is not fond");
        } catch (IOException e) {
            System.out.println("Read or write Exceptioned");
        }finally{
            if(null!=in){
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }}
            if(null!=out){
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }}}}

}
