#!/bin/bash
set -e

BASE="http://localhost:18888"
PASS=0
FAIL=0
MODELS=("gpt-5.5" "gpt-5.4" "gpt-5.4-mini" "gpt-5.3-codex" "gpt-5.3-codex-spark" "gpt-5.2" "gpt-5.2-codex" "gpt-5.1-codex" "gpt-image-2")
CHAT_MODELS=("gpt-5.5" "gpt-5.4" "gpt-5.4-mini" "gpt-5.3-codex" "gpt-5.3-codex-spark" "gpt-5.2" "gpt-5.2-codex" "gpt-5.1-codex")

green() { echo -e "\033[32m$1\033[0m"; }
red() { echo -e "\033[31m$1\033[0m"; }

run_test() {
    local name="$1"
    shift
    echo ""
    echo "=== Test: $name ==="
    if response=$("$@" 2>&1); then
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response" | head -20
        PASS=$((PASS + 1))
        green "PASS"
    else
        echo "$response" | head -10
        FAIL=$((FAIL + 1))
        red "FAIL"
    fi
}

run_python_check() {
    local name="$1"
    local script="$2"
    echo ""
    echo "=== Test: $name ==="
    if python3 - <<PY
$script
PY
    then
        PASS=$((PASS + 1))
        green "PASS"
    else
        FAIL=$((FAIL + 1))
        red "FAIL"
    fi
}

echo "Codex API Server Test Suite"
echo "============================"

# 1. Health check
run_test "Health Check" \
    curl -sf "${BASE}/health"

# 2. List models
run_test "List Models" \
    curl -sf "${BASE}/v1/models"

# 3. Validate model catalog IDs and metadata
run_python_check "Model Catalog Metadata" '
import json
import urllib.request

base = "http://localhost:18888"
models = json.load(urllib.request.urlopen(f"{base}/v1/models"))
items = {item["id"]: item for item in models["data"]}
expected_ids = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.2", "gpt-5.2-codex", "gpt-5.1-codex", "gpt-image-2"]
for model_id in expected_ids:
    assert model_id in items, f"missing model: {model_id}"

assert items["gpt-5.5"]["context_window"] == 272000
assert items["gpt-5.5"]["max_output_tokens"] == 128000
assert items["gpt-5.5"]["reasoning"] is True
assert items["gpt-5.5"]["input_modalities"] == ["text", "image"]
assert items["gpt-5.4"]["context_window"] == 1050000
assert items["gpt-5.4"]["max_output_tokens"] == 128000
assert items["gpt-5.4-mini"]["context_window"] == 400000
assert items["gpt-5.4-mini"]["max_output_tokens"] == 128000
assert items["gpt-5.4-mini"]["reasoning"] is True
assert items["gpt-5.4-mini"]["input_modalities"] == ["text", "image"]
assert items["gpt-5.3-codex-spark"]["context_window"] == 128000
assert items["gpt-5.3-codex-spark"]["max_output_tokens"] == 128000
assert items["gpt-5.2"]["context_window"] == 1050000
assert items["gpt-5.2-codex"]["context_window"] == 272000
assert items["gpt-5.1-codex"]["reasoning"] is True
assert items["gpt-image-2"]["reasoning"] is False
assert items["gpt-image-2"]["input_modalities"] == ["text", "image"]
print(json.dumps({k: items[k] for k in expected_ids}, indent=2))
'

# 4. Get specific models
for model in "${MODELS[@]}"; do
    run_test "Get Model (${model})" \
        curl -sf "${BASE}/v1/models/${model}"
done

# 5. Get non-existent model (expect 404)
echo ""
echo "=== Test: Get Non-existent Model (expect 404) ==="
status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/v1/models/nonexistent")
if [ "$status" = "404" ]; then
    green "PASS (got expected 404)"
    PASS=$((PASS + 1))
else
    red "FAIL (expected 404, got $status)"
    FAIL=$((FAIL + 1))
fi

# 6. Smoke test all advertised models
for model in "${CHAT_MODELS[@]}"; do
    run_test "Chat Completion (${model})" \
        curl -sf --max-time 60 "${BASE}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly OK\"}]}"
done

# 7. Image generation
echo ""
echo "=== Test: Image Generation (gpt-image-2) ==="
if curl -sf --max-time 180 "${BASE}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-image-2","prompt":"A tiny red square icon on a white background","size":"1024x1024","n":1,"response_format":"b64_json"}' > /tmp/codex-image-generation.json 2>/tmp/codex-image-generation.err && python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/tmp/codex-image-generation.json').read_text())
image = payload['data'][0]['b64_json']
assert isinstance(image, str) and len(image) > 1000
print(f"b64_json length: {len(image)}")
PY
then
    green "PASS"
    PASS=$((PASS + 1))
else
    python3 - <<'PY'
from pathlib import Path
for path in ('/tmp/codex-image-generation.err', '/tmp/codex-image-generation.json'):
    p = Path(path)
    if p.exists():
        print(p.read_text()[:1000])
PY
    red "FAIL"
    FAIL=$((FAIL + 1))
fi

# 8. Image edit
echo ""
echo "=== Test: Image Edit (gpt-image-2) ==="
if response=$(python3 - <<'PY'
import base64
import binascii
import json
import struct
import urllib.request
import zlib


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', binascii.crc32(kind + data) & 0xffffffff)

width = height = 8
raw = b''.join(b'\x00' + b'\xff\x00\x00' * width for _ in range(height))
png = b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + png_chunk(b'IDAT', zlib.compress(raw)) + png_chunk(b'IEND', b'')
body = json.dumps({
    "model": "gpt-image-2",
    "prompt": "Turn the red square blue",
    "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json",
}).encode()
req = urllib.request.Request(
    "http://localhost:18888/v1/images/edits",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    payload = json.load(resp)
image = payload["data"][0]["b64_json"]
assert isinstance(image, str) and len(image) > 1000
print(f"b64_json length: {len(image)}")
PY
); then
    echo "$response"
    green "PASS"
    PASS=$((PASS + 1))
else
    echo "$response" | head -5
    red "FAIL"
    FAIL=$((FAIL + 1))
fi

# 9. Streaming chat completion
echo ""
echo "=== Test: Chat Completion (streaming) ==="
response=$(curl -sN --max-time 30 "${BASE}/v1/chat/completions" \
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

# 10. Responses API image generation shorthand
echo ""
echo "=== Test: Responses API Image Generation (gpt-image-2) ==="
if curl -sf --max-time 180 "${BASE}/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-image-2","prompt":"A tiny red square icon on a white background","size":"1024x1024","quality":"medium","response_format":"b64_json","stream":false}' > /tmp/codex-responses-image.json 2>/tmp/codex-responses-image.err; then
    if python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/tmp/codex-responses-image.json').read_text())
images = [item['result'] for item in payload.get('output', []) if item.get('type') == 'image_generation_call' and item.get('result')]
assert images and len(images[0]) > 1000
print(f"b64_json length: {len(images[0])}")
PY
    then
        green "PASS"
        PASS=$((PASS + 1))
    else
        python3 - <<'PY'
from pathlib import Path
for path in ('/tmp/codex-responses-image.err', '/tmp/codex-responses-image.json'):
    p = Path(path)
    if p.exists():
        print(p.read_text()[:1000])
PY
        red "FAIL"
        FAIL=$((FAIL + 1))
    fi
else
    python3 - <<'PY'
from pathlib import Path
for path in ('/tmp/codex-responses-image.err', '/tmp/codex-responses-image.json'):
    p = Path(path)
    if p.exists():
        print(p.read_text()[:1000])
PY
    red "FAIL"
    FAIL=$((FAIL + 1))
fi

# 11. Responses API image edit shorthand
echo ""
echo "=== Test: Responses API Image Edit (gpt-image-2) ==="
if response=$(python3 - <<'PY'
import base64
import binascii
import json
import struct
import urllib.request
import zlib


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', binascii.crc32(kind + data) & 0xffffffff)

width = height = 8
raw = b''.join(b'\x00' + b'\xff\x00\x00' * width for _ in range(height))
png = b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + png_chunk(b'IDAT', zlib.compress(raw)) + png_chunk(b'IEND', b'')
body = json.dumps({
    "model": "gpt-image-2",
    "prompt": "Turn the red square blue",
    "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    "size": "1024x1024",
    "quality": "medium",
    "response_format": "b64_json",
    "stream": False,
}).encode()
req = urllib.request.Request(
    "http://localhost:18888/v1/responses",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    payload = json.load(resp)
images = [item["result"] for item in payload.get("output", []) if item.get("type") == "image_generation_call" and item.get("result")]
assert images and len(images[0]) > 1000
print(f"b64_json length: {len(images[0])}")
PY
); then
    echo "$response"
    green "PASS"
    PASS=$((PASS + 1))
else
    echo "$response" | head -5
    red "FAIL"
    FAIL=$((FAIL + 1))
fi

# 12. Logged image paths
echo ""
echo "=== Test: Logged Image Paths ==="
if python3 - <<'PY'
import json
import urllib.request
from pathlib import Path

base = 'http://localhost:18888'
logs = json.load(urllib.request.urlopen(f'{base}/v1/logs?limit=20')))
rows = logs['data']
assert rows, 'no logs returned'
def parse_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
def row_request(row):
    return parse_field(row.get('request'))
def row_response(row):
    return parse_field(row.get('response'))
image_gen = next((row for row in rows if row['endpoint'] == 'images/generations' and row_response(row).get('output_images')), None)
image_edit = next((row for row in rows if row['endpoint'] == 'images/edits' and row_request(row).get('input_images') and row_response(row).get('output_images')), None)
responses = next((row for row in rows if row['endpoint'] == 'responses' and row_response(row).get('output_images')), None)
assert image_gen is not None, 'images/generations output_images missing'
assert image_edit is not None, 'images/edits input/output images missing'
assert responses is not None, 'responses output_images missing'
for path in [
    image_gen['response']['output_images'][0]['path'],
    image_edit['request']['input_images'][0]['path'],
    image_edit['response']['output_images'][0]['path'],
    responses['response']['output_images'][0]['path'],
]:
    assert Path(path).is_file(), path
print(json.dumps({
    'images_generations_output': image_gen['response']['output_images'][0]['path'],
    'images_edits_input': image_edit['request']['input_images'][0]['path'],
    'images_edits_output': image_edit['response']['output_images'][0]['path'],
    'responses_output': responses['response']['output_images'][0]['path'],
}, indent=2))
PY
then
    green "PASS"
    PASS=$((PASS + 1))
else
    red "FAIL"
    FAIL=$((FAIL + 1))
fi

# 13. Responses API (streaming)
echo ""
echo "=== Test: Responses API (streaming) ==="
response=$(curl -sN --max-time 30 "${BASE}/v1/responses" \
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

# 14. Anthropic Messages API (non-streaming)
run_test "Anthropic Messages (non-streaming)" \
    curl -sf --max-time 60 "${BASE}/v1/messages" \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-5.4","max_tokens":100,"messages":[{"role":"user","content":"Say just the word hello"}]}'

# 15. Anthropic Messages API (streaming)
echo ""
echo "=== Test: Anthropic Messages (streaming) ==="
response=$(curl -sN --max-time 30 "${BASE}/v1/messages" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-5.4","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"Say just hi"}]}' 2>&1)
if echo "$response" | grep -q "message_start"; then
    echo "$response" | grep "event:" | head -5
    green "PASS (received Anthropic SSE events)"
    PASS=$((PASS + 1))
else
    echo "$response" | head -5
    red "FAIL (no Anthropic events received)"
    FAIL=$((FAIL + 1))
fi

# 16. Anthropic Messages with thinking
run_test "Anthropic Messages (with thinking)" \
    curl -sf --max-time 60 "${BASE}/v1/messages" \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-5.4","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":10000},"messages":[{"role":"user","content":"What is 99+1? Show reasoning."}]}'

# 17. Logs stats
run_test "Logs Stats" \
    curl -sf "${BASE}/v1/logs/stats"

# 18. Logs query
run_test "Logs Query" \
    curl -sf "${BASE}/v1/logs?limit=2"

# 19. Export JSONL (download)
echo ""
echo "=== Test: Export JSONL ==="
response=$(curl -sf "${BASE}/v1/logs/export" 2>&1)
if echo "$response" | head -1 | grep -q "request_id"; then
    echo "$response" | head -2
    green "PASS (received JSONL)"
    PASS=$((PASS + 1))
else
    echo "$response" | head -3
    red "FAIL"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "============================"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && green "All tests passed!" || red "Some tests failed!"
