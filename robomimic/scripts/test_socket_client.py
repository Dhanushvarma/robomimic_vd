# import socket

# class SimpleClient:
#     def __init__(self, host, port):
#         self.host = host
#         self.port = port

#     def connect_to_server(self):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((self.host, self.port))
#             while True:
#                 data = s.recv(1024)
#                 if not data:
#                     break
#                 print("Received data:", data.decode())
#                 print(len(data.decode()))

# if __name__ == "__main__":
#     # Replace 'server_host_ip' with the actual IP address of the Windows server
#     client = SimpleClient('192.168.103.163', 12345)
#     client.connect_to_server()

## Script for testing socket based code - Does not work
import socket
import time

class SimpleClient:
    def __init__(self, host, port, message_length):
        self.host = host
        self.port = port
        self.message_len = message_length
        self.socket = None
        self.buffer = ''

    def connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def receive_messages(self):
        if not self.socket:
            raise Exception("Not connected to server")
        try:
            while True:
                data = self.socket.recv(1024).decode()
                # print(data)
                self.buffer += data

                while True:
                    start = self.buffer.find('<REC')

                    if start != -1 and len(self.buffer) >= start + self.message_len:
                        message_to_process = self.buffer[start:start + self.message_len]
                        self.buffer = self.buffer[start + self.message_len:]
                        yield message_to_process
                    else:
                        break
        except socket.error as e:
            print(f"Connection error: {e}")


    # def get_latest_data(self):
    #     """Fetches and returns the latest complete message from the server."""
    #     if not self.socket:
    #         raise Exception("Not connected to server")

    #     try:
    #         data = self.socket.recv(4096).decode()  # Adjust the buffer size if needed
    #         self.buffer += data
    #         # return data

    #         # Process the buffer for a complete message
    #         start = self.buffer.find('<REC')
    #         if start != -1 and len(self.buffer) >= start + self.message_len:
    #             # Extract a complete message
    #             message_to_return = self.buffer[start:start + self.message_len]
    #             self.buffer = self.buffer[start + self.message_len:]
    #             return message_to_return
    #         else:
    #             # No complete message available yet
    #             return None
    #     except socket.error as e:
    #         print(f"Error while receiving latest data: {e}")
    #         return None
        

    # def get_latest_data(self):
    #     """Fetches and returns the data starting from the latest '<REC'."""
    #     if not self.socket:
    #         raise Exception("Not connected to server")

    #     try:
    #         # Read all available data from the socket
    #         flag_i = 0
    #         while True:
    #             try:
    #                 part = self.socket.recv(4096).decode()
    #                 print("Decoding Data:", flag_i)
    #                 flag_i+=1
    #                 if part:
    #                     print("Appending to buffer")
    #                     self.buffer += part
    #                 else:
    #                     break
    #             except socket.error:
    #                 print("Ingested all data")
    #                 break  # Break if no more data is available

    #         # Find the last occurrence of '<REC'
    #         last_start = self.buffer.rfind('<REC')
    #         if last_start != -1:
    #             # Extract everything from the last '<REC'
    #             message_to_return = self.buffer[last_start:]

    #             # Clear the buffer after extracting the message
    #             self.buffer = ""
    #             return message_to_return
    #         return None
    #     except socket.error as e:
    #         print(f"Error while receiving latest data: {e}")
    #         return None
            
    # def get_latest_data(self):
    #     """Fetches and returns the data starting from the latest '<REC'."""
    #     if not self.socket:
    #         raise Exception("Not connected to server")

    #     start_time = time.time()
    #     timeout_duration = 0.5  # Timeout duration in seconds

    #     try:
    #         while True:
    #             # Check for manual timeout
    #             if time.time() - start_time > timeout_duration:
    #                 print("Manual timeout occurred, breaking loop")
    #                 break

    #             try:
    #                 part = self.socket.recv(1024).decode()
    #                 if part:
    #                     self.buffer += part
    #                     start_time = time.time()  # Reset start time whenever data is received
    #                 else:
    #                     print("No more data, breaking loop")
    #                     break
    #             except socket.error as e:
    #                 print(f"Socket error: {e}")
    #                 break

    #         # Find the last occurrence of '<REC'
    #         last_start = self.buffer.rfind('<REC')
    #         if last_start != -1:
    #             message_to_return = self.buffer[last_start:]
    #             self.buffer = ""  # Clear the buffer after extracting the message
    #             print("Found ya!")
    #             return message_to_return

    #         print("Did not find ya!")
    #         print("Buffer is", self.buffer)
    #         return None
    #     finally:
    #         self.socket.settimeout(None)  # Reset the timeout to default (blocking mode)
        
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from the server.")

    def reconnect(self):
        if self.socket:
            print("Already connected.")
        else:
            self.connect_to_server()
            print("Reconnected to the server.")

if __name__ == "__main__":
    # client = SimpleClient('192.168.103.163', 5478, 130)
    # client = SimpleClient('10.25.16.82', 5478, 130)
    client = SimpleClient('192.168.1.93', 5478, 134)
    try:
        client.connect_to_server()
        # for message in client.receive_messages():
        #     # Process the message here
        #     print("Processing message:", message)

        for i in range(5):
            time.sleep(4)
            message = client.get_latest_data()
            print(message)
            

    except KeyboardInterrupt:
        client.disconnect()
