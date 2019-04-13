package socket;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class SocketServer {

    public static void main(String[] args) throws Exception {
        // 监听指定的端口
        int port = 55533;
        ServerSocket server = new ServerSocket(port);

        // server将一直等待连接的到来
        System.out.println("server将一直等待连接的到来");
        Socket socket = server.accept();  //堵塞
        // 建立好连接后，从socket中获取输入流，并建立缓冲区进行读取
        InputStream inputStream = socket.getInputStream();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));

        byte[] bytes = new byte[1024];
        int len;
        StringBuilder sb = new StringBuilder();
        //只有当客户端关闭它的输出流的时候，服务端才能取得结尾的-1

        String s = bufferedReader.readLine();
        while (s != null){
            System.out.println(s);
            sb.append(s);
            s = bufferedReader.readLine();

        }
        /*
        while ((len = inputStream.read(bytes)) != -1) {
            // 注意指定编码格式，发送方和接收方一定要统一，建议使用UTF-8

            sb.append(new String(bytes, 0, len, "UTF-8"));
        }
        */
        System.out.println("get message from client: " + sb + "ss ");

        OutputStream outputStream = socket.getOutputStream();
        outputStream.write("Hello Client,我收到了消息.".getBytes("gbk"));

        inputStream.close();
        outputStream.close();
        socket.close();
        server.close();
    }
}
