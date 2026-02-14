import argparse
import sys
import urllib.parse
import urllib.request
import http.cookiejar


def normalize_base_url(url: str) -> str:
    base = (url or "").strip()
    if not base:
        base = "http://127.0.0.1:8000"
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base.rstrip("/")


class SmokeRunner:
    def __init__(self, base_url: str, timeout: float = 20.0):
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.failures: list[str] = []
        self.passes: list[str] = []

    def _request(self, path: str, method: str = "GET", form: dict | None = None):
        url = self.base_url + path
        data = None
        headers = {}
        if form is not None:
            data = urllib.parse.urlencode(form).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
        resp = self.opener.open(req, timeout=self.timeout)
        body = resp.read().decode("utf-8", errors="replace")
        return resp.getcode(), resp.geturl(), body

    def check_status(self, name: str, path: str, expected: int = 200) -> bool:
        try:
            code, final_url, _body = self._request(path)
        except Exception as exc:
            self.failures.append(f"{name}: request failed ({exc})")
            return False
        if code != expected:
            self.failures.append(f"{name}: expected {expected}, got {code} ({final_url})")
            return False
        self.passes.append(f"{name}: OK")
        return True

    def login(self, password: str) -> bool:
        try:
            _code, final_url, _body = self._request("/login", method="POST", form={"password": password})
        except Exception as exc:
            self.failures.append(f"Login failed: {exc}")
            return False
        if "/setup" in final_url:
            self.failures.append("Login redirected to /setup. Setup is not complete yet.")
            return False
        self.passes.append("Login: OK")
        return True

    def run(self, password: str) -> int:
        # Basic availability
        if not self.check_status("Static CSS", "/static/style.css"):
            return self._finish()

        try:
            _code, final_url, body = self._request("/")
        except Exception as exc:
            self.failures.append(f"Home page failed: {exc}")
            return self._finish()

        if "/setup" in final_url or "First-time setup" in body:
            self.failures.append(
                "Setup appears incomplete. Run setup first, then re-run smoke tests."
            )
            return self._finish()

        if not self.login(password):
            return self._finish()

        checks = [
            ("Homepage", "/"),
            ("Search page", "/search"),
            ("Search results", "/search?q=love&mode=experiment"),
            ("Features page", "/features"),
            ("Rhythm", "/rhythm"),
            ("Rhythm data", "/rhythm/data"),
            ("Signals", "/rhythm/signals"),
            ("Signals data", "/rhythm/signals/data?signal=love&bin_days=30&metric=rate&series=combined&overlay=0"),
            ("Trends", "/trends"),
            ("Trends data", "/trends/data?terms=love&bin_days=30"),
            ("On This Day", "/on-this-day"),
            ("Laughs", "/laughs"),
            ("Longest Messages", "/longest-messages"),
            ("Bookmarks folders partial", "/bookmarks/folders"),
        ]
        for name, path in checks:
            self.check_status(name, path)

        return self._finish()

    def _finish(self) -> int:
        print("== Smoke Test Results ==")
        for p in self.passes:
            print(f"[PASS] {p}")
        for f in self.failures:
            print(f"[FAIL] {f}")
        if self.failures:
            print(f"\nSmoke test failed with {len(self.failures)} issue(s).")
            return 1
        print("\nSmoke test passed.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Text Archive alpha smoke tests against a running local server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the running app.")
    parser.add_argument("--password", required=True, help="App password to use for login during tests.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout seconds.")
    args = parser.parse_args()

    runner = SmokeRunner(base_url=args.base_url, timeout=args.timeout)
    return runner.run(password=args.password)


if __name__ == "__main__":
    raise SystemExit(main())
