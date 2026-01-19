#!/usr/bin/env python3
"""Generate a secure secret key for the application."""

import secrets

if __name__ == "__main__":
    secret = secrets.token_urlsafe(32)
    print(f"Generated SECRET_KEY:\n{secret}")
    print("\nAdd this to your .env file:")
    print(f"SECRET_KEY={secret}")
