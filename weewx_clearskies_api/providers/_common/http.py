"""HTTP client wrapper for provider modules (ADR-038 §3).

Each provider module instantiates ONE ProviderHTTPClient at module-load time
(not per-request).  The client owns:
  - Timeouts (connect/read/write/pool)
  - TLS verification (default ON, never disable)
  - User-Agent header injection
  - Retry/backoff for transient errors
  - Error-class translation to the canonical taxonomy

Retry behaviour (brief §confirmed-call #1, committed reason below):
  max_retries=2 means up to 3 total attempts (initial + 2 retries).
  base=0.5s, factor=2.0, cap=5.0s, with +/- 25% jitter.
  Rationale for these values: NWS at 5 req/s budget; with 5-min TTL we make
  ~1 request per 5 minutes under normal conditions.  2 retries is enough to
  survive a transient API hiccup without burning the quota or adding excessive
  latency.  0.5s base is courteous to NWS's informal rate limit.  Jitter is
  added to avoid thundering-herd in multi-worker deploys (ADR-017 §Worker).

No follow_redirects: provider modules opt-in if their docs say redirects are
normal.  Default-off prevents token-leak via accidental 30x redirect.

IPv4/IPv6: httpx uses Python's socket.getaddrinfo natively — both families
resolve correctly.  URL construction never embeds bare IPv6 literals (NWS
hostname is api.weather.gov); rules/coding.md §1 still applies.
"""

from __future__ import annotations

import logging
import random
import time

import httpx

from weewx_clearskies_api.providers._common.errors import (
    KeyInvalid,
    ProviderProtocolError,
    QuotaExhausted,
    TransientNetworkError,
)

logger = logging.getLogger(__name__)

# Retry backoff parameters.
# These values are chosen for NWS use (brief §confirmed-call #1):
#   - base=0.5s: courteous to NWS's informal rate limit
#   - factor=2.0: standard exponential backoff
#   - cap=5.0s: avoids adding excessive latency under failure
#   - jitter_fraction=0.25: +/-25% jitter to avoid thundering-herd
_BACKOFF_BASE = 0.5
_BACKOFF_FACTOR = 2.0
_BACKOFF_CAP = 5.0
_JITTER_FRACTION = 0.25


def _backoff_seconds(attempt: int) -> float:
    """Compute jittered exponential backoff for retry attempt `attempt` (0-indexed).

    attempt=0 → base * factor^0 = base = 0.5s (± jitter)
    attempt=1 → base * factor^1 = 1.0s (± jitter)
    attempt=2 → base * factor^2 = 2.0s (± jitter)  [capped at 5.0s]
    """
    delay = min(_BACKOFF_BASE * (_BACKOFF_FACTOR**attempt), _BACKOFF_CAP)
    jitter = delay * _JITTER_FRACTION * (2 * random.random() - 1)  # ±25%
    return max(0.0, delay + jitter)


class ProviderHTTPClient:
    """Sync HTTP client wrapped around httpx.Client.

    Owns: timeouts, TLS verification, User-Agent injection, retry/backoff,
    error-class translation to the canonical taxonomy.

    Each provider module instantiates ONE of these at module-load time,
    NOT per-request.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        domain: str,
        user_agent: str,
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
        max_retries: int = 2,
    ) -> None:
        self.provider_id = provider_id
        self.domain = domain
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=5.0,
                pool=5.0,
            ),
            # TLS cert verification — never disable (security-baseline §3.1).
            # NWS uses standard CA-signed certs; httpx default works.
            verify=True,
            # http2=False: NWS doesn't use HTTP/2; keep simple.
            # Re-evaluate per provider in future rounds.
            http2=False,
            # follow_redirects=False: default-off prevents token-leak via
            # accidental 30x redirect.  Provider modules opt-in explicitly.
            follow_redirects=False,
        )

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        log_url: str | None = None,
    ) -> httpx.Response:
        """Perform a GET with retry on transient errors.

        Translates upstream exception classes to canonical taxonomy:
          - httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError,
            5xx response → TransientNetworkError (after retries)
          - 429 → QuotaExhausted (with Retry-After if header present)
          - 401, 403 → KeyInvalid
          - other 4xx → ProviderProtocolError (unexpected client-side error)

        No retries on 4xx — they are not transient.

        Args:
            url: The full URL to request.
            params: Optional query-string parameters dict.
            headers: Optional additional request headers.
            log_url: When provided, used in place of `url` for all log
                messages.  Use this when `url` embeds credentials in the
                path (Aeris path-credential pattern) to prevent key leakage
                in log output (LC-E, 3b-15).  Does NOT affect the actual
                HTTP request — `url` is always the real URL sent to the server.
        """
        # Resolved URL for logging — redacted when caller provides log_url.
        _log_url = log_url if log_url is not None else url

        last_exc: Exception | None = None
        last_response: httpx.Response | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = _backoff_seconds(attempt - 1)
                logger.debug(
                    "Provider retry %d/%d after %.2fs backoff",
                    attempt,
                    self._max_retries,
                    delay,
                    extra={
                        "provider_id": self.provider_id,
                        "domain": self.domain,
                        "url": _log_url,
                    },
                )
                time.sleep(delay)

            start_ms = time.monotonic()
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.ConnectError as exc:
                # DNS failure, TCP refused, TLS handshake failure.
                # Catch specifically per rules/coding.md §3.
                logger.warning(
                    "Provider ConnectError (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                    extra={"provider_id": self.provider_id, "domain": self.domain},
                )
                last_exc = exc
                continue
            except httpx.ReadTimeout as exc:
                # Server accepted connection but didn't respond in time.
                logger.warning(
                    "Provider ReadTimeout (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                    extra={"provider_id": self.provider_id, "domain": self.domain},
                )
                last_exc = exc
                continue
            except httpx.RemoteProtocolError as exc:
                # Server violated the HTTP protocol.
                logger.warning(
                    "Provider RemoteProtocolError (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                    extra={"provider_id": self.provider_id, "domain": self.domain},
                )
                last_exc = exc
                continue
            except httpx.WriteError as exc:
                # Request write failed (rare but possible on network flap).
                logger.warning(
                    "Provider WriteError (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                    extra={"provider_id": self.provider_id, "domain": self.domain},
                )
                last_exc = exc
                continue

            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            status = response.status_code

            logger.info(
                "Provider HTTP %d %s",
                status,
                _log_url,
                extra={
                    "provider_id": self.provider_id,
                    "domain": self.domain,
                    "url": _log_url,
                    "status_code": status,
                    "elapsed_ms": elapsed_ms,
                },
            )

            # --- 4xx: do NOT retry ---
            if status == 429:
                retry_after = response.headers.get("Retry-After")
                retry_seconds: int | None = None
                if retry_after:
                    try:
                        retry_seconds = int(retry_after)
                    except ValueError:
                        retry_seconds = 60  # default if header value is unparseable
                else:
                    retry_seconds = 60  # polite default when NWS doesn't say
                raise QuotaExhausted(
                    f"Provider {self.provider_id} returned 429 Too Many Requests",
                    provider_id=self.provider_id,
                    domain=self.domain,
                    retry_after_seconds=retry_seconds,
                    status_code=status,
                )

            if status in (401, 403):
                raise KeyInvalid(
                    f"Provider {self.provider_id} returned {status} (auth failure)",
                    provider_id=self.provider_id,
                    domain=self.domain,
                    status_code=status,
                )

            if 400 <= status < 500:
                # Log the response body at ERROR before raising so operators
                # can diagnose the 400-class failure. Many providers carry a
                # useful explanation in the body (Open-Meteo: {"error": true,
                # "reason": "..."}; Aeris: {"success": false, "error": {...}};
                # NWS: {"title": "...", "detail": "..."}). Body is truncated to
                # 500 chars to avoid log bloat. NOTE: when a future round adds
                # a keyed forecast provider (Aeris/OWM/Wunderground), audit
                # whether 4xx response bodies could echo back auth credentials
                # — if so, the redaction filter or this log-body step needs an
                # extension. Open-Meteo is keyless; this is safe today.
                body_excerpt = response.text[:500] if response.text else ""
                logger.error(
                    "Provider %s 4xx %d body: %s",
                    self.provider_id,
                    status,
                    body_excerpt,
                    extra={
                        "provider_id": self.provider_id,
                        "domain": self.domain,
                        "url": _log_url,
                        "status_code": status,
                        "body_excerpt": body_excerpt,
                    },
                )
                detail = f"Provider {self.provider_id} returned unexpected {status}"
                if body_excerpt:
                    detail = f"{detail}: {body_excerpt[:200]}"
                raise ProviderProtocolError(
                    detail,
                    provider_id=self.provider_id,
                    domain=self.domain,
                    status_code=status,
                )

            # --- 5xx: retry ---
            if status >= 500:
                logger.warning(
                    "Provider 5xx %d (attempt %d/%d)",
                    status,
                    attempt + 1,
                    self._max_retries + 1,
                    extra={"provider_id": self.provider_id, "domain": self.domain},
                )
                last_response = response
                last_exc = None
                continue

            # Success (2xx/3xx — we set follow_redirects=False so 3xx is
            # returned to the caller; only 2xx is truly successful here).
            return response

        # All retries exhausted.
        if last_exc is not None:
            raise TransientNetworkError(
                f"Provider {self.provider_id} network error after "
                f"{self._max_retries + 1} attempts: {last_exc}",
                provider_id=self.provider_id,
                domain=self.domain,
            ) from last_exc

        # last_response is a 5xx — retries exhausted on server-side errors.
        assert last_response is not None  # guaranteed by loop logic above
        raise TransientNetworkError(
            f"Provider {self.provider_id} returned {last_response.status_code} "
            f"after {self._max_retries + 1} attempts",
            provider_id=self.provider_id,
            domain=self.domain,
            status_code=last_response.status_code,
        )

    def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        self._client.close()
