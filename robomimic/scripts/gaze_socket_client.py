import threading
import socket

class SimpleClient:
    def __init__(self, host, port, message_length):
        self.host = host  #HOST IP Address
        self.port = port  #PORT Number of the socket
        self.message_len = message_length  # Expected message length, #TODO: communicate length using header
        self.socket = None  # Socket Object
        self.buffer = ''  #Buffer to Store the incoming messages
        self.latest_message = None  # To store the latest message
        self.lock = threading.Lock()  # Threading to ensure decoding happens in the background unaffected
        self.running = False  # Flag

    def connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.running = True
        threading.Thread(target=self.receive_messages, daemon=True).start()  # Thread to keep the messages decoding running in the background

    def receive_messages(self):
        if not self.socket:
            raise Exception("Not connected to server")
        try:
            while self.running:
                data = self.socket.recv(1024).decode()  # Decoding messages, 1024 is specific to our gaze sensor
                self.buffer += data  # Adding the decoded data to buffer

                while True:
                    start = self.buffer.find('<REC')  # Finding the XML specific to gaze data

                    if start != -1 and len(self.buffer) >= start + self.message_len:  # Check to see if we found the XML tag (Sepcific to our gaze sensor)
                        message_to_process = self.buffer[start:start + self.message_len]  # Extracting the message of specified length from the "start"
                        self.buffer = self.buffer[start + self.message_len:]  # Emptying the buffer

                        # Store the latest message in a thread-safe way
                        with self.lock:
                            self.latest_message = message_to_process
                    else:
                        break
        except socket.error as e:
            print(f"Connection error: {e}")

    def get_latest_message(self):
        with self.lock:
            return self.latest_message

    def disconnect(self):
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from the server.")

# Usage
if __name__ == "__main__":
    client = SimpleClient('192.168.1.93', 5478, 102) # IPv4 Address of the windows machine, Port number to match, Length of the expected message
    client.connect_to_server()
    import time

    for i in range(5):
        time.sleep(3)

        latest_message = client.get_latest_message()
        print(type(latest_message))
        print(len(latest_message))
        print("Latest message:", latest_message)

    client.disconnect()