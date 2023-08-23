# Secure Image Sharing Server

## Author 
Isaac Monroy

## Project Description
This project consists of two main parts: a server that handles secure image sharing with multiple clients, and a certificate generator for the server. The server uses Elliptic Curve Diffie-Hellman (ECDH) for secure key exchange and can decrypt and save images sent by clients. It's designed to handle each client connection in a separate thread and uses SSL/TLS for secure communication.

## Libraries Used
- **socket**: For creating client-server connections.
- **threading**: For handling multiple client connections concurrently.
- **ssl**: For secure client-server communication using SSL/TLS.
- **cryptography**: For various certificate-related operations, RSA key generation, and other cryptographic tasks.
- **Crypto**: For working with Elliptic Curve keys, HKDF, SHA256, random bytes generation, HMAC, and encryption/decryption.
- **hashlib**: For hashing data.
- **datetime**: For setting the validity period of the certificate.
- **os**: For file and directory operations.

## How to Run
1. **Generate Server Certificate**: Run `server_certificate_privatekey_gen.py` to create the server's certificate and private key.
2. **Start the Server**: Run `server.py` to start listening for client connections.
3. **Client Side**: Ensure that the client-side code is configured to communicate with the server.
4. **Send Images**: Use the client application to send encrypted images to the server for processing.

## Input and Output
### Input
- **server.py**: Accepts client connections and handles the secure exchange of keys and images.
- **server_certificate_privatekey_gen.py**: No specific input is required, generates certificate and private key files for the server.

### Output
- **server.py**: Saves encrypted and decrypted images received from clients and prints out the HMAC of the decrypted image.
- **server_certificate_privatekey_gen.py**: Saves the server's certificate and private key to disk in PEM format.
