"""Sliding-window rate limiter, sized for the Gemini free tier."""

import time


class RateLimiter:
    """5 requests/minute by default; batches book several slots at once."""

    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: list[float] = []

    def _clean_old_requests(self):
        """Remove requests outside the sliding window"""
        cutoff = time.time() - self.window_seconds
        self.request_times = [t for t in self.request_times if t > cutoff]

    def get_available_slots(self) -> int:
        """Get number of requests that can be made right now"""
        self._clean_old_requests()
        return max(0, self.max_requests - len(self.request_times))

    def can_proceed(self) -> tuple[bool, int]:
        """Check if at least one request can proceed. Returns (can_proceed, wait_time_seconds)"""
        self._clean_old_requests()

        if len(self.request_times) < self.max_requests:
            return True, 0

        # Calculate wait time until oldest request expires
        oldest = min(self.request_times)
        wait_time = int(oldest + self.window_seconds - time.time()) + 1
        return False, max(0, wait_time)

    def record_request(self):
        """Record that a request was made"""
        self.request_times.append(time.time())

    def record_requests(self, count: int):
        """Record multiple requests at once (for batch processing)"""
        now = time.time()
        for _ in range(count):
            self.request_times.append(now)

    def get_status(self) -> dict:
        """Get current rate limit status"""
        self._clean_old_requests()
        available = self.max_requests - len(self.request_times)

        if available > 0:
            return {
                "ready": True,
                "wait_seconds": 0,
                "available_slots": available,
                "requests_in_window": len(self.request_times)
            }

        # Calculate wait time
        oldest = min(self.request_times) if self.request_times else time.time()
        wait_time = int(oldest + self.window_seconds - time.time()) + 1

        return {
            "ready": False,
            "wait_seconds": max(0, wait_time),
            "available_slots": 0,
            "requests_in_window": len(self.request_times)
        }


rate_limiter = RateLimiter(max_requests=5, window_seconds=60)
