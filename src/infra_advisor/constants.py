"""Shared time-base constants.

A billing month is modeled as exactly 30 days so that token-volume months
(daily tokens × DAYS_PER_MONTH) and GPU-hour months (hourly rate × HOURS_PER_MONTH)
share a single consistent basis: HOURS_PER_MONTH == DAYS_PER_MONTH × 24. This keeps
the cloud-API and self-hosted GPU columns in the same report tables directly
comparable.

Note: cloud providers commonly bill 730 h/mo (365×24/12). We use 720 h so the token
and GPU columns reconcile exactly; the ~1.4% difference is immaterial for the
directional estimates this tool produces.
"""

DAYS_PER_MONTH = 30
HOURS_PER_MONTH = DAYS_PER_MONTH * 24  # 720
