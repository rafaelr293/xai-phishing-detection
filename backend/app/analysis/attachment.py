from pathlib import PurePath

from app.domain.schemas import DetectorResultData, Indicator, ParsedEmail

HIGH_RISK_EXTENSIONS = {".exe", ".dll", ".scr", ".com", ".cpl", ".msi", ".hta", ".lnk"}
SCRIPT_EXTENSIONS = {".js", ".jse", ".vbs", ".vbe", ".ps1", ".bat", ".cmd", ".wsf"}
MACRO_EXTENSIONS = {".docm", ".dotm", ".xlsm", ".xltm", ".pptm", ".ppsm"}
WEB_EXTENSIONS = {".html", ".htm", ".svg"}
SAFE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".txt", ".csv"}
RECOGNIZED_EXTENSIONS = HIGH_RISK_EXTENSIONS | SCRIPT_EXTENSIONS | MACRO_EXTENSIONS | WEB_EXTENSIONS | SAFE_EXTENSIONS | {".zip", ".docx", ".xlsx", ".pptx", ".rtf"}


def _indicator(indicator_type: str, description: str, severity: str, evidence: dict) -> Indicator:
    return Indicator(type=indicator_type, description=description, severity=severity, evidence=evidence)


def _risk_level(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


class AttachmentAnalyzer:
    name = "attachment_analysis"

    def analyze(self, email: ParsedEmail) -> DetectorResultData:
        if not email.attachments:
            return DetectorResultData(
                detector=self.name,
                status="COMPLETED",
                risk_score=0,
                confidence=0.99,
                evidence={"attachment_count": 0, "risk_level": "low"},
            )

        indicators: list[Indicator] = []
        score = 0
        confidence_penalty = 0.0
        verification_failures: list[dict] = []
        for attachment in email.attachments:
            filename = attachment.filename.strip()
            path = PurePath(filename)
            extension = path.suffix.lower()
            suffixes = [suffix.lower() for suffix in path.suffixes]

            if extension not in RECOGNIZED_EXTENSIONS:
                confidence_penalty += 0.15
                verification_failures.append(
                    {"type": "unrecognized_extension", "filename": filename, "extension": extension or "none"}
                )

            if attachment.detected_type == "unknown":
                confidence_penalty += 0.10
                verification_failures.append(
                    {"type": "unknown_signature", "filename": filename}
                )

            if extension in HIGH_RISK_EXTENSIONS:
                score += 75
                indicators.append(
                    _indicator(
                        "dangerous_extension",
                        f"Attachment uses a high-risk executable extension: {extension}",
                        "high",
                        {"filename": filename, "extension": extension},
                    )
                )
            elif extension in SCRIPT_EXTENSIONS:
                score += 65
                indicators.append(
                    _indicator(
                        "script_attachment",
                        f"Attachment contains a script-capable extension: {extension}",
                        "high",
                        {"filename": filename, "extension": extension},
                    )
                )
            elif extension in MACRO_EXTENSIONS:
                score += 55
                indicators.append(
                    _indicator(
                        "macro_enabled_document",
                        f"Attachment is a macro-enabled Office document: {extension}",
                        "medium",
                        {"filename": filename, "extension": extension},
                    )
                )
            elif extension in WEB_EXTENSIONS:
                score += 35
                indicators.append(
                    _indicator(
                        "web_attachment",
                        f"Attachment can contain active web content: {extension}",
                        "medium",
                        {"filename": filename, "extension": extension},
                    )
                )

            if len(suffixes) >= 2 and suffixes[-1] in HIGH_RISK_EXTENSIONS | SCRIPT_EXTENSIONS:
                score += 45
                indicators.append(
                    _indicator(
                        "double_extension",
                        "Attachment filename contains a suspicious executable double extension",
                        "high",
                        {"filename": filename, "extensions": suffixes},
                    )
                )

            expected_type = {
                ".pdf": "pdf",
                ".png": "png",
                ".jpg": "jpeg",
                ".jpeg": "jpeg",
                ".gif": "gif",
            }.get(extension)
            if expected_type and attachment.detected_type not in {expected_type, "unknown"}:
                score += 55
                confidence_penalty += 0.15
                verification_failures.append(
                    {
                        "type": "signature_mismatch",
                        "filename": filename,
                        "expected_type": expected_type,
                        "detected_type": attachment.detected_type,
                    }
                )
                indicators.append(
                    _indicator(
                        "file_signature_mismatch",
                        "Attachment content does not match its filename extension",
                        "high",
                        {
                            "filename": filename,
                            "expected_type": expected_type,
                            "detected_type": attachment.detected_type,
                        },
                    )
                )

            if attachment.detected_type == "windows_executable":
                score += 90
                indicators.append(
                    _indicator(
                        "executable_signature",
                        "Attachment has a Windows executable file signature",
                        "critical",
                        {"filename": filename, "detected_type": attachment.detected_type},
                    )
                )

            nested_risky = [
                member
                for member in attachment.archive_members
                if PurePath(member).suffix.lower() in HIGH_RISK_EXTENSIONS | SCRIPT_EXTENSIONS
            ]
            if nested_risky:
                score += 75
                indicators.append(
                    _indicator(
                        "archive_contains_script",
                        "Archive contains executable or script-capable files",
                        "high",
                        {"filename": filename, "members": nested_risky[:10]},
                    )
                )

            if extension in SAFE_EXTENSIONS and attachment.size_bytes == 0:
                score += 20
                indicators.append(
                    _indicator(
                        "empty_attachment",
                        "Attachment has a safe-looking extension but contains no data",
                        "low",
                        {"filename": filename},
                    )
                )

        score = min(score, 100)
        confidence = max(0.50, round(0.95 - min(confidence_penalty, 0.45), 2))
        return DetectorResultData(
            detector=self.name,
            status="COMPLETED",
            risk_score=score,
            confidence=confidence,
            indicators=indicators,
            evidence={
                "attachment_count": len(email.attachments),
                "risk_level": _risk_level(score),
                "verification_failures": verification_failures,
            },
        )
