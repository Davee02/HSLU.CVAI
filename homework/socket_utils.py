import socket
import time

def send_command(command, host, port):
    try:
        # Create a socket connection
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        
        # Send the command
        client.send(command.encode('utf8'))
        
        # Close the connection
        client.close()
        
        # Small delay to prevent command flooding
        time.sleep(0.1)
        
        return True
    except Exception as e:
        print(f"Error sending command: {e}")
        return False