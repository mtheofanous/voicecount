# core/mailer.py
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Optional, Sequence, Tuple


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def send_smtp_email(
    *,
    to: Sequence[str],
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
    cc: Optional[Sequence[str]] = None,
    bcc: Optional[Sequence[str]] = None,
    reply_to: Optional[str] = None,
    attachments: Optional[Sequence[Tuple[str, bytes, str]]] = None,
) -> None:
    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT", "587"))
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    sender = _env("SMTP_FROM", user)
    reply_to = (reply_to or _env("SMTP_REPLY_TO") or "").strip() or None

    if not host or not user or not password:
        raise RuntimeError("SMTP is not configured (SMTP_HOST/SMTP_USER/SMTP_PASSWORD).")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join([x for x in to if x])
    if cc:
        msg["Cc"] = ", ".join([x for x in cc if x])
    if reply_to:
        msg["Reply-To"] = reply_to

    # Important: include BCC recipients only in the SMTP envelope, not headers
    recipients = list(to) + list(cc or []) + list(bcc or [])

    # Plain text + optional HTML alternative
    msg.set_content(text_body or "")
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    # Optional attachments
    for att in attachments or []:
        try:
            filename, data, mime = att
        except Exception:
            continue

        ctype = (mime or "application/octet-stream").split("/", 1)
        maintype = ctype[0] if ctype else "application"
        subtype = ctype[1] if len(ctype) > 1 else "octet-stream"
        msg.add_attachment(data or b"", maintype=maintype, subtype=subtype, filename=filename or "attachment")

    # STARTTLS
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(user, password)
        server.send_message(msg, from_addr=user, to_addrs=recipients)
