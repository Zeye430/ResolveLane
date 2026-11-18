#!/bin/sh
# Publishes the demo accounts for the sign-in menu, only when RESOLVELANE_DEMO_SIGN_IN=true.
# Anyone who can open the page can then sign in as any of these accounts: keep it for a local
# demo with invented accounts, never for a server others can reach.
set -eu
target=/usr/share/nginx/html/demo-accounts.json
rm -f "$target"
if [ "${RESOLVELANE_DEMO_SIGN_IN:-false}" = "true" ] && [ -n "${RESOLVELANE_API_TOKENS:-}" ]; then
    printf '%s' "$RESOLVELANE_API_TOKENS" > "$target"
    echo "demo sign-in is on: the sign-in page offers the accounts in RESOLVELANE_API_TOKENS"
fi
