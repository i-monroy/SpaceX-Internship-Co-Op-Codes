# Secure Image Sharing Client

## Author 
Isaac Monroy

## Project Description
This project consists of two main parts: a client that securely sends images to a server, and a certificate generator for the client. The client uses Elliptic Curve Diffie-Hellman (ECDH) for secure key exchange and can encrypt images to send to the server. It's designed to work in conjunction with a server that handles secure image sharing with multiple clients.

## Libraries Used
- **socket**: For creating client-server connections.
- **ssl**: For secure client-server communication using SSL/TLS.
- **cryptography**: For various certificate-related operations, RSA key generation, and other cryptographic tasks.
- **Crypto**: For working with Elliptic Curve keys, HKDF, SHA256, random bytes generation, HMAC, and encryption/decryption.
- **os**: For file and directory operations.

## How to Run
1. **Generate Client Certificate**: Run `client_certificate_privatekey_gen.py` to create the client's certificate and private key.
2. **Start the Client**: Run `client.py` to start the client and prepare for secure communication with the server.
3. **Server Side**: Ensure that the server-side code is configured to communicate with the client.
4. **Send Images**: Use the client application to send encrypted images to the server for processing.

## Input and Output
### Input
- **client.py**: Accepts user registration, login credentials, and image file selection for secure transmission.
- **client_certificate_privatekey_gen.py**: No specific input is required, generates certificate and private key files for the client.

### Output
- **client.py**: Sends encrypted images to the server, handles user registration, login, and provides feedback via the graphical user interface.
- **client_certificate_privatekey_gen.py**: Saves the client's certificate and private key to disk in PEM format.
