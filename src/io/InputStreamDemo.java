package io;

import java.io.*;

public class InputStreamDemo {
    //字节流的操作

    public void InputStreamTest(){
        try {
            //InputStream用于读取基于字节的数据，一次读取一个字节，这是一个InputStream的例子：
            //字节流需要强转char
            System.out.println("in InputStream : ");
            InputStream inputStream = new FileInputStream("D:\\io.txt");
            int date =  inputStream.read();
            while (date != -1){
                System.out.print((char) date);
                date = inputStream.read();
            }
            System.out.print("\n");
            inputStream.close();
        }catch (Exception e){
            System.out.println(e.getMessage() + " error");
        }
    }

    public void InputStreamWithBuffer(){
        try {
            InputStream inputStream = new FileInputStream("D:\\io.txt");
            byte[] buffer = new byte[1024];
            int data = inputStream.read(buffer);
            while (data!=-1){
                for (int i = 0; i < data; i++)
                    System.out.print((char) buffer[i]);
                data = inputStream.read(buffer);
            }

        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }

    public void OutputTest() {
        try {
            byte bWrite[] = { 11, 21, 3, 40, 5 };
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            outputStream.write("hello world!!".getBytes());

            byte[] buf = outputStream.toByteArray();
            for (int i = 0; i < 11; i++)
                System.out.print((char)buf[i]);

        } catch (IOException e) {
            System.out.print(e.getMessage());
        }

    }

    public static void main(String[] args){
        InputStreamDemo inputStreamDemo = new InputStreamDemo();
        inputStreamDemo.InputStreamTest();
        System.out.println("----------");
        inputStreamDemo.InputStreamWithBuffer();
        System.out.println("----------");
        inputStreamDemo.OutputTest();

    }

}
