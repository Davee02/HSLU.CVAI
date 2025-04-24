import socket
    
class SocketClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None
        
    def connect(self):
        try:
            # Create a socket connection
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Error connecting to server: {e}")
    
    def send(self, command):
        if self.client:
            try:
                # Send the command
                self.client.send(command.encode('utf8'))
                print(f"Sent command: {command}")
            except Exception as e:
                print(f"Error sending command: {e}")
    
    def close(self):
        if self.client:
            self.client.close()
            print("Connection closed")