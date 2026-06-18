# Bug Fix: JWT Session Persistence Across Server Restarts

**Date:** 2026-06-18  
**Project:** OMyFish (Streamlit + FastAPI, local dev + HF Spaces)  
**Time spent:** ~3 hours on what looked like a 5-minute task

---

## STAR Summary

### Situation

OMyFish has a JWT-based login system. Every time the Streamlit app restarted, users had to log in again. On Hugging Face Spaces this was especially painful because the Space idles and restarts frequently. The app was already generating a `JWT_SECRET` — the expectation was that fixing persistence would be straightforward.

### Task

Make the login session survive:
1. Page refreshes (browser F5)
2. Full server restarts (`make app` killed and relaunched)

### Action

What looked like one problem turned out to be three independent failures, discovered in layers:

**Attempt 1 — `extra-streamlit-components` CookieManager**  
The natural choice: store the token in a browser cookie. Installed the library, wired up `_cm.set()` on login and `_cm.get()` on restore.  
*Failure:* The `set()` call was silently failing — I passed `max_age=` but the function signature expects `expires_at=` (a datetime). Python swallowed the `TypeError` because the call was inside a `@st.fragment`. Even after fixing the parameter, the CookieManager is a React component that needs one extra render cycle before its data is available in Python — so reading on cold start always returned `None`.

**Attempt 2 — `st.query_params`**  
Store the token in the URL (`?token=...`). Simpler, no third-party library. On login: `st.query_params["token"] = jwt_token`. On cold load: read it back and restore the session.  
*Partial success:* Page refresh worked — the token stayed in the URL. But server restart failed — when Streamlit reconnects after a server kill, the browser navigates back to the base URL, stripping the query params.

**Attempt 3 — `.local_session` file**  
Write the token to a local file on login, read it back on startup. Combined with query params (for refresh), this should cover both scenarios.  
*Failure:* The file was created correctly, but after a restart the token couldn't be decoded. The `jwt.decode()` call was silently failing — swallowed by a broad `except Exception: pass` — and clearing the file.

**Root cause discovery**  
Added a quick diagnostic that printed `settings.jwt_secret[:8]` on startup. The prefix changed on every restart. The `JWT_SECRET` environment variable was never actually being loaded into the app process.

The variable was set in `.env`, and `~/.bashrc` had `export $(grep -v '^#' .env | xargs)` to load it. This works in an **interactive terminal** — but `make app` launched by the Bash tool runs in a **non-interactive shell** that does not source `~/.bashrc`. So every startup called `secrets.token_hex(32)` and generated a fresh secret, making every saved token immediately invalid.

**The actual fix (2 lines)**  
Added `python-dotenv` to `shared/config.py`:

```python
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")
```

This loads `.env` directly at import time, regardless of how the process was started. JWT_SECRET is now consistent across all launches.

### Result

Login now persists across:
- ✅ Page refresh (via `st.query_params`)
- ✅ Full server restart (via `.local_session` file + consistent `JWT_SECRET`)
- ✅ HF Spaces (fixed secret set in Space secrets, no file needed)

The two-line `python-dotenv` fix was the actual solution. Everything before it was treating symptoms.

---

## Lessons

**1. Broad `except` blocks are landmines.**  
Both the cookie write failure and the JWT decode failure were swallowed by `except Exception: pass`. Each one masked the real problem and sent the investigation in the wrong direction. Log or re-raise; don't silently swallow.

**2. "It's in my environment" is not the same as "it's in the app's environment."**  
`~/.bashrc` is for interactive shells. Any app launched by a script, cron, IDE, or tool gets a non-interactive shell and won't see it. The correct place to load env vars is in the app itself (via `python-dotenv` or equivalent), not in the shell profile.

**3. Validate your storage primitive before building on it.**  
Before wiring up a cookie-based auth flow, spend 30 seconds verifying that `_cm.set()` actually writes a cookie to the browser. A quick check in DevTools → Application → Cookies would have saved an hour.

**4. Debug the layer you think is working, not just the layer that's failing.**  
The session file approach looked correct. The bug was one layer below it — in the config module that fed the secret to the JWT decoder. When a chain of operations silently fails, start by verifying each link independently.
