
import socket
import subprocess
import os
import time

def find_free_port():
    """Find a free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    _, port = s.getsockname()
    s.close()
    return port

def launch_tensorboard(events_folder, port):
    """Launch TensorBoard with the specified port."""
    tb_cmd = f"tensorboard --logdir={events_folder} --port={port}"
    subprocess.Popen(tb_cmd, shell=True)

def main(events_folder):
    """Main function."""
    
    if not os.path.exists(events_folder):
        print("Events folder does not exist!")
        os.makedirs(events_folder)
    
    port = find_free_port()
    print(f"Selected port: {port}")
    launch_tensorboard(events_folder, port)
    time.sleep(5)