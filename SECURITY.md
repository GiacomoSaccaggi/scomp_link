# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.1.x   | ✅ Active support   |
| 1.0.x   | ⚠️ Critical fixes only |
| < 1.0   | ❌ End of life      |

## Reporting a Vulnerability

If you discover a security vulnerability in scomp-link, please report it responsibly.

**DO NOT** open a public GitHub issue for security vulnerabilities.

### How to Report

1. Email: **giacomo.saccaggi@gmail.com**
2. Subject: `[SECURITY] scomp-link — <brief description>`
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days
- **Fix release**: within 30 days for critical issues

### Scope

The following are in scope:
- Arbitrary code execution (e.g., via malicious `.scomp` files)
- Dependency vulnerabilities
- Path traversal in file operations
- Information disclosure

## Known Security Considerations

### Pickle Deserialization (`.scomp` files)

⚠️ **WARNING**: The `.scomp` format uses Python's `pickle` module to serialize models and preprocessors. Loading a `.scomp` file from an untrusted source can execute arbitrary code.

**Only load `.scomp` files you created yourself or received from a trusted source.**

```python
# SAFE: loading your own artifact
artifact = ScompArtifact.load("my_model.scomp")

# DANGEROUS: loading untrusted files
artifact = ScompArtifact.load("downloaded_from_internet.scomp")  # ⚠️ RISK
```

We are evaluating safer serialization alternatives (safetensors, ONNX) for future releases.
