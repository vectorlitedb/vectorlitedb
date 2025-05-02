# Security Policy

## Supported Versions

VectorLiteDB is currently in alpha development. Once we reach our first stable release, this section will be updated with information about which versions receive security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of VectorLiteDB seriously. If you believe you've found a security vulnerability, please follow these steps:

### Where to Report

Please **DO NOT** report security vulnerabilities through public GitHub issues.

Instead, please report them via email to [security@vectorlite.tech](mailto:security@vectorlite.tech). If possible, encrypt your message with our PGP key (to be provided in the future).

### What to Include

When reporting a vulnerability, please include as much information as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Affected versions
- Any possible mitigations or workarounds
- If available, proof-of-concept code or screenshots

### Response Process

After you report a vulnerability:

1. A project maintainer will acknowledge receipt of the report within 48 hours
2. We will confirm the vulnerability and determine its impact
3. We will develop and test a fix
4. We will prepare a security advisory and release a patch
5. We will publicly disclose the vulnerability after a fix is available

### Disclosure Policy

- We follow a coordinated disclosure process
- The reporter will be kept informed throughout the process
- We aim to release fixes within 30 days of report verification
- Public disclosure will typically occur after a fix is available

## Security Best Practices for VectorLiteDB Users

### Data Sensitivity

While VectorLiteDB focuses on providing privacy-first, on-device vector storage, please be mindful of the type of data you store:

- Avoid storing Personally Identifiable Information (PII) in vector metadata if possible
- Consider encrypting sensitive metadata using your own encryption before storage
- Be aware that anyone with access to the `.db` file will have access to the stored vectors and metadata

### File Security

- Protect access to your VectorLiteDB database files using appropriate file system permissions
- Consider using encrypted storage for the directory containing your database files
- Implement appropriate backup procedures for your database files

## Security Measures in VectorLiteDB

VectorLiteDB implements several security features:

- Data integrity checks through checksums
- Corruption detection and recovery mechanisms
- Safe file handling practices

## Acknowledgments

We would like to thank the following individuals who have responsibly disclosed security issues to us:

*This section will be updated as security researchers contribute.*

## Questions

If you have questions about this policy or VectorLiteDB security in general, please contact [security@vectorlite.tech](mailto:security@vectorlite.tech).