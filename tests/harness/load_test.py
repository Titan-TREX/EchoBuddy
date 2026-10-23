"""Small load test against a RUNNING server (local or deployed).

    uv run python -m tests.harness.load_test http://localhost:8000 [--token T] [--chats 8]

Three checks:
  1. /health under heavy concurrency       -> latency percentiles, zero errors expected
  2. rate limiting actually engages        -> a burst past the limit must produce 429s
  3. N simultaneous real text chats        -> every one should answer (this makes real LLM and
                                              TTS calls, so keep N small)

Note the rate limit is per client IP: running part 2 first would throttle part 3, so the
chats run before the burst. Wait a minute between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import uuid

import httpx


def _percentiles(samples: list[float]) -> str:
    ordered = sorted(samples)
    pick = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))]  # noqa: E731
    return f"p50 {pick(0.50) * 1000:.0f} ms | p95 {pick(0.95) * 1000:.0f} ms | max {ordered[-1] * 1000:.0f} ms"


async def health_storm(client: httpx.AsyncClient, base: str, total: int, concurrency: int) -> bool:
    gate = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    bad = 0

    async def one() -> None:
        nonlocal bad
        async with gate:
            start = time.perf_counter()
            try:
                bad += (await client.get(f"{base}/health")).status_code != 200
            except httpx.HTTPError:
                bad += 1
            latencies.append(time.perf_counter() - start)

    started = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(total)))
    took = time.perf_counter() - started
    print(f"[1] /health x{total} @ {concurrency} concurrent: {total / took:.0f} req/s | {_percentiles(latencies)} | errors {bad}")
    return bad == 0


async def concurrent_chats(client: httpx.AsyncClient, base: str, headers: dict, count: int) -> bool:
    async def chat(i: int):
        start = time.perf_counter()
        r = await client.post(
            f"{base}/api/text-chat",
            json={"text": f"What is {i + 2} times 3?", "user_id": f"u_load{uuid.uuid4().hex[:16]}"},
            headers=headers,
            timeout=120,
        )
        return r.status_code, time.perf_counter() - start, (r.json().get("bot_text") if r.is_success else r.text[:80])

    results = await asyncio.gather(*(chat(i) for i in range(count)), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception) and r[0] == 200]
    times = [r[1] for r in ok]
    print(f"[3] {count} simultaneous real chats: {len(ok)} answered"
          + (f" | {_percentiles(times)}" if times else ""))
    for r in results[:3]:
        print("     sample:", r if isinstance(r, Exception) else r[2])
    return len(ok) == count


async def rate_limit_engages(client: httpx.AsyncClient, base: str, headers: dict, burst: int) -> bool:
    codes = [(await client.get(f"{base}/api/docs", headers={**headers, "X-User-Id": "u_loadtest0000000001"})).status_code
             for _ in range(burst)]
    limited = codes.count(429)
    print(f"[2] burst of {burst} guarded requests: {codes.count(200)} allowed, {limited} rate-limited (429)")
    return limited > 0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--token", default="")
    parser.add_argument("--chats", type=int, default=8)
    parser.add_argument("--burst", type=int, default=60)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as client:
        results = [
            await health_storm(client, base, total=500, concurrency=50),
            await concurrent_chats(client, base, headers, args.chats),
            await rate_limit_engages(client, base, headers, args.burst),
        ]
    print("\nRESULT:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
