"""
Author: Isaac Monroy
Title: Secure Image Sharing Server
Description:
    This algorithm sets up the server side of a secure image sharing application. 
    It can handle multiple client connections and decrypts and saves an image sent 
    by each client. 

    It uses Elliptic Curve Diffie-Hellman (ECDH) for secure key exchange, obtaining 
    a shared symmetric key between the server and each client. This symmetric key is 
    then used to decrypt the image file sent by the client, ensuring the confidentiality 
    of the image data.

    The server is set up to handle each client connection in a separate thread, allowing 
    for simultaneous processing of multiple clients. 
    
    Secure communication is implemented with SSL/TLS.
"""

# Networking and multi-threading libraries
import socket  # Create client-server connection
import threading  # Handle multiple client connections concurrently

# Cryptography and security related libraries
import ssl  # Secure client-server connection using SSL/TLS
from Crypto.PublicKey import ECC  # Generate and work with Elliptic Curve keys
from Crypto.Protocol.KDF import HKDF  # Derive a symmetric key from the ECDH shared secret
from Crypto.Hash import SHA256  # Hash data for the HKDF and other operations
from Crypto.Random import get_random_bytes  # Generate random bytes for cryptographic operations
import hmac  # Create a hashed message authentication code (HMAC)
import hashlib  # Hash data for the HMAC
from cryptography.fernet import Fernet  # Generate encryption keys and encrypt/decrypt messages


# Set this to False to disable secure communication
SECURE_COMMUNICATION_ENABLED = True  

# Paths for server's certificate and private key used for SSL/TLS communication
CERTIFICATE_PATH = './server_cert_and_privkey/certificate.pem'
PRIVATE_KEY_PATH = './server_cert_and_privkey/private_key.pem'

def receive_all(sock, size):
    """
    Receives data from a socket.

    This function ensures that all the data sent from the client is received 
    completely by continuing to read from the socket.
    """
    data = b''
    while len(data) < size:
        more = sock.recv(size - len(data))
        if not more:
            raise IOError('Failed to receive all data')
        data += more
    return data

def handle_client(c, addr):
    """
    Handles client connections in separate threads.

    For each client, it receives the client's public key, computes a shared 
    secret, derives a symmetric key, and then receives, decrypts, and saves 
    an image sent by the client.
    """
    print('Connected by', addr)

    # Generate a new ECC key
    key = ECC.generate(curve='P-256')

    # Send the public key to the client
    pub_key = key.public_key().export_key(format='DER')
    c.sendall(len(pub_key).to_bytes(4, 'big'))
    c.sendall(pub_key)

    # Receive the client's public key
    client_pub_key_len = int.from_bytes(c.recv(4), 'big')
    client_pub_key = ECC.import_key(c.recv(client_pub_key_len))

    # Generate the shared secret using server's private key and client's public key
    shared_secret = key.d * client_pub_key.pointQ

    # Use a key derivation function (KDF) to get the symmetric encryption key from the shared secret
    symmetric_key = HKDF(
        master=shared_secret.x.to_bytes((int(shared_secret.x).bit_length() + 7) // 8, 'big'),
        key_len=32, 
        salt=b'', 
        hashmod=SHA256
    )

    # Receive the symmetric encryption key from the client
    key_size = int.from_bytes(c.recv(4), 'big')
    key = c.recv(key_size)

    # Send an acknowledgement to the client
    c.sendall(b'ACK')

    # Receive the size of the image from the client
    size = int.from_bytes(c.recv(4), 'big')

    # Receive the encrypted image from the client
    encrypted_image = receive_all(c, size)
    
    # Save the encrypted image
    with open('server_encrypted_image.jpg', 'wb') as f:
        f.write(encrypted_image)
        
    # Decrypt the image using the symmetric encryption key
    cipher_suite = Fernet(key)
    decrypted_image = cipher_suite.decrypt(encrypted_image)

    # Calculate and print the HMAC of the decrypted image
    h = hmac.new(key, decrypted_image, hashlib.sha256)
    print(f"HMAC of image on server side: {h.hexdigest()}")

    # Save the decrypted image
    with open('server_decrypted_image.jpg', 'wb') as f:
        f.write(decrypted_image)
    
    print('Image received and decrypted successfully.')
    c.close()
    
def create_secure_socket(sock):
    """
    Wraps a socket with SSL/TLS for secure communication.

    The function loads the server's certificate and private key and wraps 
    the provided socket to secure its communications.
    """
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=CERTIFICATE_PATH, keyfile=PRIVATE_KEY_PATH)
    
    return context.wrap_socket(sock, server_side=True)

def server():
    """
    Starts the server.

    It creates a socket, binds it to an address, and listens for incoming 
    connections.
    """
    HOST = '127.0.0.1'  
    PORT = 65432        

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    if SECURE_COMMUNICATION_ENABLED:
        s = create_secure_socket(s)
    
    s.bind((HOST, PORT))
    s.listen()

    while True:
        c, addr = s.accept()
        thread = threading.Thread(target=handle_client, args=(c, addr))
        thread.start()

# Start the server
server()