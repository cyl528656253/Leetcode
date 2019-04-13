package socket;

import java.io.*;
import java.net.Socket;

public class SocketClient {

    public static  void main(String[] args) throws IOException {
        String host = "127.0.0.1";
        int port = 55533;
        // 与服务端建立连接
        Socket socket = new Socket(host, port);
        // 建立连接后获得输出流
        OutputStream outputStream = socket.getOutputStream();
        String message = "你好  cai 这是来自  客户端的信息";
        socket.getOutputStream().write(message.getBytes("gbk"));
        //通过shutdownOutput高速服务器已经发送完数据，后续只能接受数据
        socket.shutdownOutput();

        InputStream inputStream = socket.getInputStream();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        byte[] bytes = new byte[1024];
        int len;
        StringBuilder sb = new StringBuilder();

        String s = bufferedReader.readLine();
        while (s != null){
            sb.append(s);
            s = bufferedReader.readLine();

        }

        /*
        while ((len = inputStream.read(bytes)) != -1) {
            //注意指定编码格式，发送方和接收方一定要统一，建议使用UTF-8
            sb.append(new String(bytes, 0, len,"UTF-8"));
        }
        */
        System.out.println("get message from server: " + sb);

        inputStream.close();
        outputStream.close();
        socket.close();

    }
}
