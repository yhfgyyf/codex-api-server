#!/bin/bash
set -e

BASE="http://localhost:18888"
PASS=0
FAIL=0

green() { echo -e "\033[32m$1\033[0m"; }
red() { echo -e "\033[31m$1\033[0m"; }

test_endpoint() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "=== Test: $name ==="
    if response=$(eval "$cmd" 2>&1); then
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response" | head -20
        PASS=$((PASS + 1))
        green "PASS"
    else
        echo "$response" | head -10
        FAIL=$((FAIL + 1))
        red "FAIL"
    fi
}

echo "Codex API Server Test Suite"
echo "============================"

# 1. Health check
test_endpoint "Health Check" \
    "curl -sf $BASE/health"

# 2. List models
test_endpoint "List Models" \
    "curl -sf $BASE/v1/models"

# 3. Get specific model
test_endpoint "Get Model (gpt-5.4)" \
    "curl -sf $BASE/v1/models/gpt-5.4"

# 4. Get non-existent model (expect 404)
echo ""
echo "=== Test: Get Non-existent Model (expect 404) ==="
status=$(curl -s -o /dev/null -w "%{http_code}" $BASE/v1/models/nonexistent)
if [ "$status" = "404" ]; then
    green "PASS (got expected 404)"
    PASS=$((PASS + 1))
else
    red "FAIL (expected 404, got $status)"
    FAIL=$((FAIL + 1))
fi

# 5. Non-streaming chat completion
test_endpoint "Chat Completion (non-streaming)" \
    "curl -sf --max-time 60 $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"gpt-5.4\",\"messages\":[{\"role\":\"user\",\"content\":\"Say just the word hello\"}]}'"

# 6. Streaming chat completion
echo ""
echo "=== Test: Chat Completion (streaming) ==="
response=$(curl -sN --max-time 30 $BASE/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"Say just the word hi"}],"stream":true}' 2>&1)
if echo "$response" | grep -q "data:"; then
    echo "$response" | head -5
    green "PASS (received SSE data)"
    PASS=$((PASS + 1))
else
    echo "$response" | head -5
    red "FAIL (no SSE data received)"
    FAIL=$((FAIL + 1))
fi

# 7. Responses API (streaming)
echo ""
echo "=== Test: Responses API (streaming) ==="
response=$(curl -sN --max-time 30 $BASE/v1/responses \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-5.4","instructions":"You are a helpful assistant.","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hey"}]}],"stream":true}' 2>&1)
if echo "$response" | grep -q "response.output_text.delta"; then
    echo "$response" | grep "output_text.delta" | head -3
    green "PASS (received Responses API events)"
    PASS=$((PASS + 1))
else
    echo "$response" | head -5
    red "FAIL (no Responses events received)"
    FAIL=$((FAIL + 1))
fi

# 8. Chat with system message
test_endpoint "Chat with System Message" \
    "curl -sf --max-time 60 $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"gpt-5.4\",\"messages\":[{\"role\":\"system\",\"content\":\"Always respond in French\"},{\"role\":\"user\",\"content\":\"Say hello\"}]}'"

echo ""
echo "============================"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && green "All tests passed!" || red "Some tests failed!"
