# 🔒 Security Policy

GeoVision takes the security and integrity of our Earth observation platform, inference microservices, and geospatial data processing pipelines seriously.

---

## 1. Supported Versions

Security updates and critical vulnerability patches are applied to the following versions:

| Version | Supported          | Security Patch Status |
| :---    | :---               | :---                  |
| `1.0.x` | :white_check_mark: | Actively supported    |
| `< 1.0` | :x:                | End of Life (EOL)     |

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability in GeoVision, please do not open a public issue.

1. **Email Notification**: Send a detailed vulnerability report to the core security team at `security@geovision.ai` or submit a private security advisory on GitHub.
2. **Details to Include**:
   - Component affected (`geovision.api`, `geovision.geo`, web client, docker container).
   - Step-by-step instructions or proof-of-concept payload to reproduce the issue.
   - Potential impact on system confidentiality, integrity, or availability.
3. **Response Window**: We acknowledge security advisories within 48 hours and provide remediation timelines within 7 business days.

---

## 3. Environment & Secret Management

- **API Keys & Model Credentials**: Never commit `.env` files, MLflow credentials, OpenAI/VLM tokens, or cloud storage secrets into version control.
- **Pre-Commit Hooks**: GeoVision enforces `pre-commit` secret scanning via `detect-secrets` and `gitleaks` configurations.
- **Docker Production Images**: Production images run under non-root user execution contexts (`appuser`) with minimal filesystem write permissions.
