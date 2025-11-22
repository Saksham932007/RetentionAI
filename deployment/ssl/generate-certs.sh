#!/bin/bash
# Generate self-signed SSL certificates for development/testing
# For production, replace with proper CA-signed certificates

set -euo pipefail

# Certificate details
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORGANIZATION="RetentionAI"
ORGANIZATIONAL_UNIT="IT Department"
COMMON_NAME="localhost"
EMAIL="admin@retentionai.local"

# Certificate files
KEY_FILE="key.pem"
CERT_FILE="cert.pem"

echo "Generating self-signed SSL certificate..."
echo "Common Name: $COMMON_NAME"

# Generate private key
openssl genrsa -out "$KEY_FILE" 2048

# Generate certificate signing request
openssl req -new -key "$KEY_FILE" -out "csr.pem" -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORGANIZATION/OU=$ORGANIZATIONAL_UNIT/CN=$COMMON_NAME/emailAddress=$EMAIL"

# Generate self-signed certificate
openssl x509 -req -days 365 -in "csr.pem" -signkey "$KEY_FILE" -out "$CERT_FILE" -extensions v3_ca -extfile <(
    echo "[v3_ca]"
    echo "subjectAltName=@alt_names"
    echo "[alt_names]"
    echo "DNS.1=localhost"
    echo "DNS.2=retentionai.local"
    echo "DNS.3=*.retentionai.local"
    echo "IP.1=127.0.0.1"
    echo "IP.2=0.0.0.0"
)

# Clean up CSR file
rm "csr.pem"

# Set proper permissions
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo "SSL certificate generated successfully!"
echo "Certificate: $CERT_FILE"
echo "Private key: $KEY_FILE"
echo ""
echo "Note: This is a self-signed certificate for development/testing only."
echo "For production, use proper CA-signed certificates."

# Display certificate info
echo ""
echo "Certificate details:"
openssl x509 -in "$CERT_FILE" -text -noout | head -20