"""
Author: Isaac Monroy
Title: Secure Image Sharing Client
Description:
    This algorithm creates the client side of a secure image sharing application.
    The client is able to register a new user, login with existing credentials and
    send an encrypted image to a server over a secure SSL/TLS connection.

    The algorithm includes a graphical user interface (GUI) built with tkinter. 
    The GUI prompts the user to input their username and password and to select an 
    action, either registration or login. 

    For secure communication, the algorithm uses the Elliptic Curve Diffie-Hellman 
    (ECDH) key exchange protocol to generate a shared symmetric key between the client
    and server. This symmetric key is used to encrypt the image file with the Fernet 
    symmetric encryption (AES) before sending it to the server.

    For secure storage of user credentials, the script employs the Werkzeug library's 
    password hashing function.
"""
# Network, communication, and security related libraries
import socket  # Create client-server connection
import ssl  # Secure client-server connection using SSL/TLS
from Crypto.PublicKey import ECC  # Generate and work with Elliptic Curve keys
from Crypto.Protocol.KDF import HKDF  # Derive a symmetric key from the ECDH shared secret
from Crypto.Hash import SHA256  # Hash data for the HKDF and other operations
from Crypto.Random import get_random_bytes  # Generate random bytes for cryptographic operations
import hmac  # Create a hashed message authentication code (HMAC)
import hashlib  # Hash data for the HMAC
from cryptography.fernet import Fernet  # Generate encryption keys and encrypt/decrypt messages

# User management and file handling libraries
import os # Interact with the operating system
import json  # Store and retrieve user data from a file
from werkzeug.security import generate_password_hash, check_password_hash  # Hash and verify passwords

# User Interface libraries
import tkinter as tk  # Build the graphical user interface (GUI)
from tkinter import filedialog  # Prompt user to select a file via the GUI


# Set to False to disable secure communication. Default is True.
SECURE_COMMUNICATION_ENABLED = True  

# Paths for client certificate and private key
CERTIFICATE_PATH = './client_cert_and_privkey/certificate.pem'
PRIVATE_KEY_PATH = './client_cert_and_privkey/private_key.pem'

# Path for the server certificate
SERVER_CERTIFICATE_PATH = './server_cert_and_privkey/certificate.pem'

# The hostname of the server to connect to
SERVER_HOSTNAME = 'localhost'

def register_gui():
    """
    Registers a new user.

    If the user chooses to register, this function collects and stores the 
    credentials given by the user in a new JSON file. If the file exists, 
    the new credentials are added.
    """
    # Collect username and password input from the user
    username = username_entry.get()
    password = password_entry.get()
    
    # Hash the password for secure storage
    hashed_password = generate_password_hash(password)

    # Generate a new encryption key and convert it to a string for JSON storage
    key = Fernet.generate_key()
    key_str = key.decode()

    # If the users file exists, load it. Otherwise, create a new dictionary.
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    
    # If the username already exists, abort registration
    if username in users:
        print("Username already exists.")
        return

    # Save the hashed password and key to the users dictionary
    users[username] = {'password': hashed_password, 'key': key_str}

    # Write the updated users dictionary back to the file
    with open('users.json', 'w') as f:
        json.dump(users, f)

    print("User registered successfully.")

def login(username, password):    
    """
    Logs in an existing user.
    
    This function checks the provided username and password in the 
    stored credentials. If they match, the user is logged in.
    """
    # Attempt to open the users file
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
    except FileNotFoundError:
        print("No users are registered.")
        return None

    # Check if the provided username and password match the stored credentials
    if username in users and check_password_hash(users[username]['password'], password):
        print("Logged in successfully.")
        return users[username]['key']
    else:
        print("Invalid username or password.")
        return None

def login_gui():
    """
    Handles login from the GUI.
    
    This function calls the login function with the provided username 
    and password, then updates the GUI if the login was successful.
    """
    # Get the entered username and password
    username = username_entry.get()
    password = password_entry.get()
    
    # Attempt to login with the provided credentials
    key = login(username, password)
    
    # If login was successful, show the choose file button
    if key is not None:
        choose_file_button.grid(row=6, column=0, columnspan=2)

def choose_file():
    """
    Opens a file dialog for the user to choose a file to send.
    
    If a file is chosen, this function calls the client function with the 
    chosen file's path as a parameter.
    """
    filename = filedialog.askopenfilename()
    
    # Only call the client function if a file was chosen
    if filename:
        client(filename)

def create_secure_client_socket(sock):
    """
    Creates a secure socket using SSL.

    The secure socket is created using the client certificate, private key,
    and the server certificate.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.load_cert_chain(certfile=CERTIFICATE_PATH, keyfile=PRIVATE_KEY_PATH)
    context.load_verify_locations(cafile=SERVER_CERTIFICATE_PATH)
    return context.wrap_socket(sock, server_hostname=SERVER_HOSTNAME)

def client(image_path):
    """
    The main client function.

    This function connects to the server, exchanges public keys, derives 
    a shared symmetric key, encrypts and sends an image to the server.
    """
    HOST = '127.0.0.1'  
    PORT = 65432        

    # Generate a new ECC key
    key = ECC.generate(curve='P-256')

    # Start the client socket and make it secure if required
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if SECURE_COMMUNICATION_ENABLED:
        s = create_secure_client_socket(s)

    # Connect to the server and exchange public keys
    with s:
        s.connect((HOST, PORT))

        # Receive the server's public key
        server_pub_key_len = int.from_bytes(s.recv(4), 'big')
        server_pub_key = ECC.import_key(s.recv(server_pub_key_len))

        # Send the public key to the server
        pub_key = key.public_key().export_key(format='DER')
        s.sendall(len(pub_key).to_bytes(4, 'big'))
        s.sendall(pub_key)

        # Generate the shared secret
        shared_secret = key.d * server_pub_key.pointQ

        # Use a key derivation function (KDF) to get the symmetric encryption key
        symmetric_key = HKDF(
            master=shared_secret.x.to_bytes((int(shared_secret.x).bit_length() + 7) // 8, 'big'),
            key_len=32, 
            salt=b'', 
            hashmod=SHA256
        )

        # Generating the key
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        print(f"Generated key: {key}")

        # Check if the file exists and abort if it doesn't
        if not os.path.isfile(image_path):
            print("The file does not exist.")
            return

        # Open, encrypt, and send the image
        with open(image_path , 'rb') as img_file:
            image_data = img_file.read()
            h = hmac.new(key, image_data, hashlib.sha256)
            print(f"HMAC of image on client side: {h.hexdigest()}")
            encrypted_image = cipher_suite.encrypt(image_data)

        # Send the size of the key
        s.sendall(len(key).to_bytes(4, 'big'))

        # Send the key
        s.sendall(key)

        # Wait for an acknowledgement from the server before proceeding
        ack = s.recv(1024)
        if ack.decode('utf-8') != 'ACK':
            print('Failed to send key')
            return

        # Send the size of the image
        s.sendall(len(encrypted_image).to_bytes(4, 'big'))

        # Send the encrypted image in chunks
        chunk_size = 1024
        for i in range(0, len(encrypted_image), chunk_size):
            s.sendall(encrypted_image[i:i+chunk_size])

# Create the main application window
root = tk.Tk()
root.geometry("400x400")
root.title("Secure Image Sharing")  

# Configure the window's grid
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Add labels, entries, and buttons to the window
title_label = tk.Label(root, text="Secure Image Sharing", font=("Helvetica", 16))
title_label.grid(row=0, column=0, columnspan=2, pady=10)

instruction_label = tk.Label(root, text="Please enter your credentials and select an action:", font=("Helvetica", 12))
instruction_label.grid(row=1, column=0, columnspan=2, pady=10)

username_label = tk.Label(root, text="Username:")
username_label.grid(row=2, column=0)
username_entry = tk.Entry(root)
username_entry.grid(row=2, column=1, padx=10)

password_label = tk.Label(root, text="Password:")
password_label.grid(row=3, column=0)
password_entry = tk.Entry(root, show="*")
password_entry.grid(row=3, column=1, padx=10)

note_label = tk.Label(root, text="Note: If unregistered, register first then log in with the same credentials.", wraplength=300, justify="center")
note_label.grid(row=4, column=0, columnspan=2)

login_button = tk.Button(root, text="Login", command=login_gui, bg='blue', fg='white')
login_button.grid(row=5, column=0, pady=10)

register_button = tk.Button(root, text="Register", command=register_gui, bg='green', fg='white')
register_button.grid(row=5, column=1, pady=10)

choose_file_button = tk.Button(root, text="Choose file", command=choose_file, bg='orange', fg='white')

root.grid_rowconfigure(7, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the main event loop
root.mainloop()

