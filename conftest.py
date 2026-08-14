"""Cấu hình pytest dùng chung, chủ yếu cho giới hạn biến môi trường Windows."""

from __future__ import annotations

import hashlib


def pytest_make_parametrize_id(config, val, argname):
    """Rút gọn giá trị tham số lớn trước khi pytest đưa node id vào môi trường."""
    rendered = repr(val)
    if len(rendered) <= 200:
        return None
    digest = hashlib.sha256(rendered.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{argname}-large-{digest}"
