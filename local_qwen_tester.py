import base64
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 7864
POLL_INTERVAL_SECONDS = 5
MAX_TRANSIENT_STATUS_ERRORS = 12
MAX_STATUS_RETRY_DELAY_SECONDS = 30
TRANSIENT_STATUS_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
OUTPUT_DIR = Path(__file__).resolve().parent / "local_test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen v19 Clothing Edit Tester</title>
  <style>
    :root {
      --bg: #0f1216;
      --panel: #171b22;
      --panel-2: #1f2530;
      --text: #f5f7fb;
      --muted: #9ca9bb;
      --accent: #f59e0b;
      --accent-2: #fb7185;
      --accent-3: #7dd3fc;
      --border: #2a3240;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(245,158,11,0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(251,113,133,0.14), transparent 20%),
        radial-gradient(circle at bottom right, rgba(125,211,252,0.12), transparent 24%),
        linear-gradient(180deg, #0b0f15 0%, #111726 100%);
      color: var(--text);
      min-height: 100vh;
    }
    .wrap {
      width: min(1220px, calc(100% - 32px));
      margin: 24px auto 40px;
      display: grid;
      gap: 20px;
    }
    .panel {
      background: rgba(23, 27, 34, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      backdrop-filter: blur(8px);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.26);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.08;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    form {
      display: grid;
      gap: 14px;
      margin-top: 18px;
    }
    .grid {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .preview-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      align-items: start;
    }
    label {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    input, textarea, select, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    textarea {
      min-height: 132px;
      resize: vertical;
    }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #130b08;
      font-weight: 700;
      border: none;
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .status {
      display: grid;
      gap: 10px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(125, 211, 252, 0.1);
      color: #c8f0ff;
      border: 1px solid rgba(125, 211, 252, 0.25);
      font-weight: 600;
    }
    .meta, pre {
      background: #0d121a;
      border-radius: 14px;
      border: 1px solid var(--border);
      padding: 14px;
      overflow: auto;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 360px;
    }
    .card-title {
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0.02em;
      color: #d9e1ee;
      text-transform: uppercase;
    }
    img {
      width: 100%;
      max-width: 100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #0b0f14;
      min-height: 220px;
      object-fit: contain;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Qwen v19 Clothing Edit Tester</h1>
      <p>Upload a source image, describe the clothing change, and this page will submit the request to your RunPod endpoint and poll until the final image is ready.</p>
      <form id="job-form">
        <div class="grid">
          <label>RunPod Endpoint ID
            <input name="endpoint_id" placeholder="biqd9c2lr7dqjn" required>
          </label>
          <label>RunPod API Key
            <input name="api_key" type="password" placeholder="rpa_..." required>
          </label>
        </div>
        <label>Instruction
          <textarea name="instruction" required>change the dress to red satin and remove the socks</textarea>
        </label>
        <div class="grid">
          <label>Input Image
            <input id="image-file" name="image_file" type="file" accept="image/*" required>
          </label>
          <label>Parser Model
            <select name="parser_model">
              <option value="atr" selected>ATR</option>
              <option value="lip">LIP</option>
            </select>
          </label>
          <label>Seed
            <input name="seed" type="number" value="42">
          </label>
          <label>Steps
            <input name="steps" type="number" min="1" value="8">
          </label>
          <label>CFG
            <input name="cfg" type="number" min="0.1" step="0.1" value="1.0">
          </label>
          <label>Denoise
            <input name="denoise" type="number" min="0.1" max="1" step="0.01" placeholder="auto">
          </label>
          <label>Sampler
            <input name="sampler_name" value="euler_ancestral">
          </label>
          <label>Scheduler
            <input name="scheduler" value="beta">
          </label>
          <label>Target Width
            <input name="target_width" type="number" min="256" step="64" value="1024">
          </label>
          <label>Target Height
            <input name="target_height" type="number" min="256" step="64" value="1024">
          </label>
          <label>Mask Expand Pixels
            <input name="mask_expand_pixels" type="number" min="0" value="12">
          </label>
          <label>Mask Blend Pixels
            <input name="mask_blend_pixels" type="number" min="0" value="4">
          </label>
          <label>Context Expand Factor
            <input name="context_expand_factor" type="number" min="1.0" step="0.05" value="1.2">
          </label>
          <label>Output Padding
            <select name="output_padding">
              <option value="0">0</option>
              <option value="8">8</option>
              <option value="16">16</option>
              <option value="32" selected>32</option>
              <option value="64">64</option>
              <option value="128">128</option>
              <option value="256">256</option>
              <option value="512">512</option>
            </select>
          </label>
          <label>Device Mode
            <select name="device_mode">
              <option value="gpu" selected>GPU</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label>Checkpoint Name
            <input name="checkpoint_name" value="Qwen-Rapid-AIO-NSFW-v19.safetensors">
          </label>
        </div>
        <button id="submit-btn" type="submit">Generate Image</button>
      </form>
    </div>

    <div class="panel">
      <div class="preview-grid">
        <div>
          <div class="card-title">Source</div>
          <img id="source-preview" alt="Source preview">
        </div>
        <div>
          <div class="card-title">Result</div>
          <img id="result-preview" class="hidden" alt="Result preview">
        </div>
      </div>
    </div>

    <div class="panel status">
      <div id="state-pill" class="pill">Idle</div>
      <div id="state-text" class="meta">Fill the form and submit a job.</div>
      <pre id="json-output">{}</pre>
    </div>
  </div>

  <script>
    const form = document.getElementById("job-form");
    const submitButton = document.getElementById("submit-btn");
    const statePill = document.getElementById("state-pill");
    const stateText = document.getElementById("state-text");
    const jsonOutput = document.getElementById("json-output");
    const resultPreview = document.getElementById("result-preview");
    const sourcePreview = document.getElementById("source-preview");
    const imageFileInput = document.getElementById("image-file");
    let pollHandle = null;

    function setState(label, text) {
      statePill.textContent = label;
      stateText.textContent = text;
    }

    function resetOutputs() {
      resultPreview.removeAttribute("src");
      resultPreview.classList.add("hidden");
      jsonOutput.textContent = "{}";
    }

    function showResult(data) {
      jsonOutput.textContent = JSON.stringify(data, null, 2);
      const result = data.result || {};
      if (result.image_url) {
        resultPreview.src = result.image_url;
        resultPreview.classList.remove("hidden");
      }
    }

    function readFileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Failed to read the selected file."));
        reader.readAsDataURL(file);
      });
    }

    imageFileInput.addEventListener("change", async () => {
      const file = imageFileInput.files && imageFileInput.files[0];
      if (!file) {
        sourcePreview.removeAttribute("src");
        return;
      }
      sourcePreview.src = await readFileAsDataUrl(file);
    });

    async function pollStatus(localJobId) {
      if (pollHandle) {
        clearInterval(pollHandle);
      }

      const runPoll = async () => {
        const response = await fetch(`/api/status?id=${encodeURIComponent(localJobId)}`);
        const data = await response.json();
        showResult(data);

        const label = data.state || "UNKNOWN";
        const remote = data.remote_status ? ` | RunPod: ${data.remote_status}` : "";
        setState(label, (data.message || "Waiting") + remote);

        if (["COMPLETED", "FAILED"].includes(label)) {
          clearInterval(pollHandle);
          pollHandle = null;
          submitButton.disabled = false;
        }
      };

      await runPoll();
      pollHandle = setInterval(runPoll, 3000);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submitButton.disabled = true;
      resetOutputs();
      setState("SUBMITTING", "Submitting job to the local proxy...");

      const formData = new FormData(form);
      const imageFile = formData.get("image_file");
      if (!(imageFile instanceof File) || !imageFile.size) {
        setState("FAILED", "Please choose an image file.");
        submitButton.disabled = false;
        return;
      }

      const payload = Object.fromEntries(formData.entries());
      payload.image_data_url = await readFileAsDataUrl(imageFile);
      delete payload.image_file;

      const response = await fetch("/api/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      showResult(data);

      if (!response.ok) {
        setState("FAILED", data.message || "Request failed.");
        submitButton.disabled = false;
        return;
      }

      setState(data.state || "SUBMITTED", data.message || "Job submitted.");
      await pollStatus(data.local_job_id);
    });
  </script>
</body>
</html>
"""


def _set_job(local_job_id: str, **updates) -> dict:
    with JOBS_LOCK:
        job = JOBS.setdefault(local_job_id, {})
        job.update(updates)
        return dict(job)


def _get_job(local_job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(local_job_id)
        return dict(job) if job else None


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _http_json(method: str, url: str, headers: dict | None = None, body: dict | None = None) -> dict:
    payload = None
    final_headers = dict(headers or {})
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        final_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=payload, headers=final_headers, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _save_output_bytes(local_job_id: str, remote_file: dict) -> str | None:
    data = remote_file.get("data")
    output_type = remote_file.get("type")
    filename = remote_file.get("filename") or f"{local_job_id}.bin"
    if not data:
        return None

    if output_type in {"s3_url", "bucket_url"}:
        return str(data)

    if output_type != "base64":
        return None

    output_path = OUTPUT_DIR / f"{local_job_id}_{Path(filename).name}"
    output_path.write_bytes(base64.b64decode(data))
    return f"/outputs/{output_path.name}"


def _strip_data_uri(data: str) -> str:
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        return data.split(",", 1)[1]
    return data


def _build_runpod_input(form_data: dict[str, str], image_base64: str) -> dict:
    runpod_input = {
        "instruction": form_data["instruction"],
        "image_base64": _strip_data_uri(image_base64),
        "parser_model": form_data["parser_model"],
        "seed": int(form_data["seed"]),
        "steps": int(form_data["steps"]),
        "cfg": float(form_data["cfg"]),
        "sampler_name": form_data["sampler_name"],
        "scheduler": form_data["scheduler"],
        "target_width": int(form_data["target_width"]),
        "target_height": int(form_data["target_height"]),
        "mask_expand_pixels": int(form_data["mask_expand_pixels"]),
        "mask_blend_pixels": int(form_data["mask_blend_pixels"]),
        "context_expand_factor": float(form_data["context_expand_factor"]),
        "output_padding": int(form_data["output_padding"]),
        "device_mode": form_data["device_mode"],
    }

    if form_data.get("denoise", "").strip():
        runpod_input["denoise"] = float(form_data["denoise"])
    if form_data.get("checkpoint_name", "").strip():
        runpod_input["checkpoint_name"] = form_data["checkpoint_name"].strip()

    return runpod_input


def _read_http_error_details(exc: urllib.error.HTTPError) -> str:
    details = exc.read().decode("utf-8", errors="replace").strip()
    return details or str(exc)


def _is_transient_status_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in TRANSIENT_STATUS_HTTP_CODES


def _set_transient_status_error(
    local_job_id: str,
    *,
    remote_job_id: str,
    remote_status: str,
    error_text: str,
    retry_count: int,
    retry_delay_seconds: int,
) -> None:
    _set_job(
        local_job_id,
        state="RUNNING",
        message=(
            f"RunPod status check failed temporarily and will retry in "
            f"{retry_delay_seconds}s ({retry_count}/{MAX_TRANSIENT_STATUS_ERRORS}). Last error: {error_text}"
        ),
        remote_job_id=remote_job_id,
        remote_status=remote_status,
        raw={
            "error": error_text,
            "transient_status_error": True,
            "retry_count": retry_count,
            "retry_delay_seconds": retry_delay_seconds,
        },
    )


def _process_job(local_job_id: str, endpoint_id: str, api_key: str, runpod_input: dict) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    remote_job_id = ""
    remote_status = "PENDING"
    consecutive_status_errors = 0

    try:
        submit_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        submit_response = _http_json("POST", submit_url, headers=headers, body={"input": runpod_input})
        remote_job_id = str(submit_response.get("id") or "")
        if not remote_job_id:
            raise ValueError(f"RunPod response missing job id: {submit_response}")
        remote_status = str(submit_response.get("status", "IN_QUEUE"))

        _set_job(
            local_job_id,
            state="SUBMITTED",
            message="Job accepted by RunPod.",
            remote_job_id=remote_job_id,
            remote_status=remote_status,
            raw=submit_response,
        )

        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{remote_job_id}"
        while True:
            try:
                status_response = _http_json("GET", status_url, headers=headers)
            except urllib.error.HTTPError as exc:
                error_text = f"HTTP {exc.code}: {_read_http_error_details(exc)}"
                if not _is_transient_status_http_error(exc):
                    raise

                consecutive_status_errors += 1
                if consecutive_status_errors > MAX_TRANSIENT_STATUS_ERRORS:
                    _set_job(
                        local_job_id,
                        state="FAILED",
                        message=(
                            "RunPod status checks kept failing after "
                            f"{MAX_TRANSIENT_STATUS_ERRORS} retries. Last error: {error_text}"
                        ),
                        remote_job_id=remote_job_id,
                        remote_status=remote_status,
                        raw={
                            "error": error_text,
                            "transient_status_error": True,
                            "retry_count": consecutive_status_errors,
                        },
                    )
                    return

                retry_delay_seconds = min(
                    POLL_INTERVAL_SECONDS * consecutive_status_errors,
                    MAX_STATUS_RETRY_DELAY_SECONDS,
                )
                _set_transient_status_error(
                    local_job_id,
                    remote_job_id=remote_job_id,
                    remote_status=remote_status,
                    error_text=error_text,
                    retry_count=consecutive_status_errors,
                    retry_delay_seconds=retry_delay_seconds,
                )
                time.sleep(retry_delay_seconds)
                continue
            except urllib.error.URLError as exc:
                consecutive_status_errors += 1
                error_text = f"Status poll error: {exc.reason}"
                if consecutive_status_errors > MAX_TRANSIENT_STATUS_ERRORS:
                    _set_job(
                        local_job_id,
                        state="FAILED",
                        message=(
                            "RunPod status checks kept failing after "
                            f"{MAX_TRANSIENT_STATUS_ERRORS} retries. Last error: {error_text}"
                        ),
                        remote_job_id=remote_job_id,
                        remote_status=remote_status,
                        raw={
                            "error": error_text,
                            "transient_status_error": True,
                            "retry_count": consecutive_status_errors,
                        },
                    )
                    return

                retry_delay_seconds = min(
                    POLL_INTERVAL_SECONDS * consecutive_status_errors,
                    MAX_STATUS_RETRY_DELAY_SECONDS,
                )
                _set_transient_status_error(
                    local_job_id,
                    remote_job_id=remote_job_id,
                    remote_status=remote_status,
                    error_text=error_text,
                    retry_count=consecutive_status_errors,
                    retry_delay_seconds=retry_delay_seconds,
                )
                time.sleep(retry_delay_seconds)
                continue

            consecutive_status_errors = 0
            remote_status = str(status_response.get("status", "UNKNOWN"))
            _set_job(
                local_job_id,
                state="RUNNING" if remote_status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"} else remote_status,
                message="Waiting for RunPod to finish the job.",
                remote_job_id=remote_job_id,
                remote_status=remote_status,
                raw=status_response,
            )

            if remote_status == "COMPLETED":
                output = status_response.get("output", {})
                result: dict[str, str] = {}
                images = output.get("images") or []

                if images:
                    image_url = _save_output_bytes(local_job_id, images[0])
                    if image_url:
                        result["image_url"] = image_url

                _set_job(
                    local_job_id,
                    state="COMPLETED",
                    message="Image generation finished.",
                    remote_job_id=remote_job_id,
                    remote_status=remote_status,
                    raw=status_response,
                    result=result,
                )
                return

            if remote_status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                error_text = status_response.get("error") or status_response.get("message") or "RunPod job failed."
                _set_job(
                    local_job_id,
                    state="FAILED",
                    message=str(error_text),
                    remote_job_id=remote_job_id,
                    remote_status=remote_status,
                    raw=status_response,
                )
                return

            time.sleep(POLL_INTERVAL_SECONDS)
    except urllib.error.HTTPError as exc:
        details = _read_http_error_details(exc)
        _set_job(
            local_job_id,
            state="FAILED",
            message=f"HTTP {exc.code}: {details}",
            remote_job_id=remote_job_id,
            remote_status=remote_status,
            raw={"error": details},
        )
    except Exception as exc:  # noqa: BLE001
        _set_job(
            local_job_id,
            state="FAILED",
            message=str(exc),
            remote_job_id=remote_job_id,
            remote_status=remote_status,
            raw={"error": str(exc)},
        )


class QwenTesterHandler(BaseHTTPRequestHandler):
    server_version = "QwenLocalTester/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/status"):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            local_job_id = params.get("id", [""])[0]
            job = _get_job(local_job_id)
            if not job:
                _json_response(self, {"message": "Unknown local job id."}, status=404)
                return
            _json_response(self, {"local_job_id": local_job_id, **job})
            return

        if self.path.startswith("/outputs/"):
            filename = self.path.removeprefix("/outputs/")
            file_path = OUTPUT_DIR / Path(filename).name
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Output file not found.")
                return
            payload = file_path.read_bytes()
            mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        if self.path != "/api/submit":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            endpoint_id = str(payload.get("endpoint_id", "")).strip()
            api_key = str(payload.get("api_key", "")).strip()
            instruction = str(payload.get("instruction", "")).strip()
            image_data_url = str(payload.get("image_data_url", "")).strip()

            if not endpoint_id:
                raise ValueError("Endpoint ID is required.")
            if not api_key:
                raise ValueError("API key is required.")
            if not instruction:
                raise ValueError("Instruction is required.")
            if not image_data_url:
                raise ValueError("Please choose an image file.")

            form_data = {
                "instruction": instruction,
                "parser_model": str(payload.get("parser_model", "atr")),
                "seed": str(payload.get("seed", "42")),
                "steps": str(payload.get("steps", "8")),
                "cfg": str(payload.get("cfg", "1.0")),
                "denoise": str(payload.get("denoise", "")),
                "sampler_name": str(payload.get("sampler_name", "euler_ancestral")),
                "scheduler": str(payload.get("scheduler", "beta")),
                "target_width": str(payload.get("target_width", "1024")),
                "target_height": str(payload.get("target_height", "1024")),
                "mask_expand_pixels": str(payload.get("mask_expand_pixels", "12")),
                "mask_blend_pixels": str(payload.get("mask_blend_pixels", "4")),
                "context_expand_factor": str(payload.get("context_expand_factor", "1.2")),
                "output_padding": str(payload.get("output_padding", "32")),
                "device_mode": str(payload.get("device_mode", "gpu")),
                "checkpoint_name": str(payload.get("checkpoint_name", "")),
            }

            runpod_input = _build_runpod_input(form_data, image_data_url)
            local_job_id = uuid.uuid4().hex
            _set_job(
                local_job_id,
                state="QUEUED",
                message="Local proxy accepted the job and is sending it to RunPod.",
                remote_status="PENDING",
                result={},
                raw={},
            )

            worker = threading.Thread(
                target=_process_job,
                args=(local_job_id, endpoint_id, api_key, runpod_input),
                daemon=True,
            )
            worker.start()

            _json_response(
                self,
                {
                    "local_job_id": local_job_id,
                    "state": "QUEUED",
                    "message": "Job queued locally. Polling will begin automatically.",
                    "result": {},
                    "raw": {},
                },
            )
        except json.JSONDecodeError:
            _json_response(self, {"message": "Invalid JSON payload."}, status=400)
        except ValueError as exc:
            _json_response(self, {"message": str(exc)}, status=400)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), QwenTesterHandler)
    print(f"Qwen tester running at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
