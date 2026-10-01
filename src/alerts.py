"""ส่งการแจ้งเตือน: log + เขียนลง logs/alerts.jsonl + webhook (Discord/Slack) ถ้าตั้งค่าไว้"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx

logger = logging.getLogger("alerts")


def send_alert(title: str, detail: dict | str, severity: str = "warning") -> dict:
    alert = {
        "ts": datetime.now(UTC).isoformat(),
        "severity": severity,
        "title": title,
        "detail": detail,
    }
    logger.warning("🚨 [ALERT][%s] %s | %s", severity.upper(), title, detail)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "alerts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(alert, ensure_ascii=False) + "\n")

    webhook = os.getenv("ALERT_WEBHOOK_URL")
    if webhook:
        text = f"🚨 [{severity.upper()}] {title}\n```{json.dumps(detail, ensure_ascii=False)[:1500]}```"
        try:
            # Discord ใช้ "content", Slack ใช้ "text" -> ส่งทั้งคู่
            httpx.post(webhook, json={"content": text, "text": text}, timeout=5)
        except httpx.HTTPError as e:
            logger.error("ส่ง webhook ไม่สำเร็จ: %s", e)
    return alert
