"""
Author: Isaac Monroy
Title: Certificate & Private Key Generator: Server
Description:
    This script generates a self-signed certificate and corresponding private 
    key for a server.

    The RSA algorithm is used to generate the key pair. A self-signed certificate 
    is then created, using the key pair and other information such as issuer name, 
    subject name, and validity period.

    The certificate and private key are saved to disk in PEM format. The files can
    then be used in the server script to create a secure SSL/TLS connection.
"""

# Cryptography-related libraries
from cryptography import x509  # Various certificate related operations
from cryptography.x509.oid import NameOID  # Predefined values for creating a certificate
from cryptography.hazmat.primitives import hashes  # Hashing algorithms
from cryptography.hazmat.primitives.asymmetric import rsa  # RSA key generation
from cryptography.hazmat.primitives import serialization  # Convert keys and certificates to bytes for storage

# Standard Python libraries
import datetime  # For setting the validity period of the certificate
import os  # For file and directory operations

def generate_self_signed_cert(cert_path, private_key_path):
    """
    Generate a self-signed certificate.

    This function generates an RSA private key and a self-signed certificate. 
    It then saves the private key and certificate to the specified paths.
    """
    # Generate an RSA private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Save the private key to the specified path
    with open(private_key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Define the subject and issuer details of the certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"My Company"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])

    # Define the validity period of the certificate
    valid_from = datetime.datetime.utcnow()
    valid_to = valid_from + datetime.timedelta(days=10)

    # Define the certificate serial number
    serial_number = x509.random_serial_number()

    # Define the public key for the certificate (the public part of the RSA key pair)
    public_key = key.public_key()

    # Create and sign the certificate
    builder = x509.CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)
    builder = builder.not_valid_before(valid_from)
    builder = builder.not_valid_after(valid_to)
    builder = builder.serial_number(serial_number)
    builder = builder.public_key(public_key)
    builder = builder.add_extension(
        x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
        critical=False,
    )

    certificate = builder.sign(
        private_key=key, algorithm=hashes.SHA256()
    )

    # Save the certificate to the specified path
    with open(cert_path, "wb") as f:
        f.write(certificate.public_bytes(serialization.Encoding.PEM))

# Define the paths where the certificate and private key will be saved for the server
path = "./server_cert_and_privkey"
if not os.path.exists(path):
    os.mkdir(path)
certificate_path = os.path.join(path, 'certificate.pem')
private_key_path = os.path.join(path, 'private_key.pem')

# Generate a self-signed certificate and save it, along with the private key
generate_self_signed_cert(certificate_path, private_key_path)

