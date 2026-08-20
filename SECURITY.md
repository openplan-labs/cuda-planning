# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a vulnerability

cuda-planning compiles bundled CUDA C sources at runtime via NVRTC and
executes them on the local GPU; it does not open network connections or
execute user-supplied code. If you believe you have found a security
issue — for example in how kernel sources are loaded, or a memory
safety problem exploitable through crafted inputs — please report it
privately:

- Email **erwin.lejeune15@gmail.com** with a description, a minimal
  reproduction, and the affected version.
- Or use GitHub's private vulnerability reporting on this repository
  (Security → Report a vulnerability).

You will get an acknowledgement within a week. Please do not open a
public issue for suspected vulnerabilities before a fix is released.
