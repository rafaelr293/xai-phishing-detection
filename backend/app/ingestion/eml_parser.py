import hashlib
import re
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

from app.domain.schemas import AttachmentData, ParsedEmail

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _part_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def parse_eml(path: Path, max_size_bytes: int) -> ParsedEmail:
    if path.suffix.lower() != ".eml":
        raise ValueError("Only .eml files can be ingested")
    size = path.stat().st_size
    if size > max_size_bytes:
        raise ValueError(f"Email exceeds the {max_size_bytes}-byte size limit")

    raw = path.read_bytes()
    message = BytesParser(policy=policy.default).parsebytes(raw)
    plain_text = ""
    html_body = ""
    attachments: list[AttachmentData] = []

    for part in message.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename or part.get_content_disposition() == "attachment":
            attachments.append(
                AttachmentData(
                    filename=filename or "unnamed",
                    content_type=part.get_content_type(),
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
        elif part.get_content_type() == "text/plain":
            plain_text += _part_text(part)
        elif part.get_content_type() == "text/html":
            html_body += _part_text(part)

    body_for_urls = f"{plain_text}\n{html_body}"
    urls = sorted(set(URL_PATTERN.findall(body_for_urls)))
    headers = {key: str(value) for key, value in message.items()}
    recipients = [address.strip() for address in (message.get_all("To", []) + message.get_all("Cc", [])) if address.strip()]

    return ParsedEmail(
        content_hash=hashlib.sha256(raw).hexdigest(),
        source_path=str(path),
        sender=message.get("From"),
        recipients=recipients,
        subject=message.get("Subject"),
        headers=headers,
        plain_text=plain_text,
        html_body=html_body,
        urls=urls,
        attachments=attachments,
    )
