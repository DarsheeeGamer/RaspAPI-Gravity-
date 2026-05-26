# Gravity AI API documentation

---




---



### 1. Bearer Token Authorization
All request endpoints (except telemetry `/api/metrics` and interactive `/docs`) require an `Authorization: Bearer grav_<api_key>` token in the header.

### 2. IP-based Rate Limiting
To prevent abuse, the gateway implements a strict client IP rate-limiting policy (`request.client.host` key):
*   **Default Shared Pool:** **20 requests per hour** per client IP.
*   **Custom Linked Pool:** **2000 requests per hour** per client IP.

### 3. Lifetime Default allowance
The shared gateway pool restricts users to a **hard cap of 20 lifetime requests** per provider per API key. This encourages users to register their own accounts for continued high-volume usage.

---

## 📡 Gateway API Endpoints

### 1. `POST /v1/chat/completions`
An openai compatible endpoint which can be used with the openai lib.
#### Request Headers
```http
Authorization: Bearer grav_<api_key>
Content-Type: application/json
```

#### Request Body with tools
```json 
{
  "model": "cursor/auto",
  "messages": [
    { "role": "system", "content": "You are a coding assistant." },
    { "role": "user", "content": "Fetch the weather for New York." }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Fetch current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" }
          }
        }
      }
    }
  ],
  "stream": false
}
```

#### Response with tool calls
```json
{
  "id": "chatcmpl-3044e0dde5f079fd2b359cd3",
  "object": "chat.completion",
  "created": 1779830686,
  "model": "cursor/auto",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "tool_0aa17d12-6fad-4b83-b3b0-085cd2390be",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"New York\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

---

### 2. `POST /api/generate-key`
Generates a new API key
```http
POST /api/generate-key
```
#### Response
```json
{
  "apikey": "grav_9eCxHF2vF9qasHh1aAZD5dpX-pPOqiwZ",
  "status": "active",
  "pool": "default"
}
```

---

### 3. `POST /api/add`
Allows you to link a account with the specific API Key. Doing so transitions your usage limits immediately to the **Custom linked Pool (2000 req/hr)**.

#### Request Headers
```http
Authorization: Bearer grav_<api_key>
Content-Type: application/json
```

#### Request Body Payload
```json
{
  "account_id": "accountid",
  "provider": "windsurf",
  "api_key": "add windsurf or cursor token. signout of windsurf IDE and then login.it will show token starting with sum ott . then paste this here. i will need to add the CURSOR auth implementation to this WEBAPI. the gravity cli does support it tho..",
  "pool": "default",
  "priority": 100,
  "weight": 1,
  "transport": {
    "timeout_ms": 120000
  }
}
```

#### Field Schema Reference
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `account_id` | `string` | **Yes** | A unique tenant identifier for the registered credential. |
| `provider` | `string` | **Yes** | Target service provider (`cursor`, `windsurf`, or `antigravity`). |
| `api_key` | `string` | **Yes** | The raw token credential (Cursor refresh token, Windsurf Firebase token, or Antigravity API secret). |
| `pool` | `string` | No | Target pool class routing (defaults to `default`). |
| `priority` | `integer` | No | Priority ordering within the tenant account group (defaults to `100`). |
| `weight` | `integer` | No | Load balancing selection weight (defaults to `1`). |
| `transport.timeout_ms` | `integer` | No | Request timeout configuration in milliseconds (defaults to `120000`). |

#### Response
```json
{
  "status": "success",
  "message": "Account successfully linked to API key"
}
```

---

### 4. `GET /api/metrics`
Telemetry for the public to show the total number of requests the server has handled and also the most used model
```http
GET /api/metrics
```
#### Response
```json
{
  "total_requests": 14,
  "most_used_model": "cursor/auto",
  "status": "healthy"
}
```

---

### Note on payloads which have tools:
Tools calls are not handled on the server side since the server currently does not have execution capabilities unlike the native GravityV2 package. So instead it sends the tool calls back to you, the client and you will have to execute it in your environment and send the tool result back to the API to continue with the response generation

To run the loop:
1. Client submits prompt + tools array.
2. Gateway outputs a response containing `finish_reason: "tool_calls"`.
3. Client intercepts the `tool_calls` list and executes the function locally.
4. Client appends the assistant's request and the execution output (`role: "tool"`, containing `tool_call_id`) back to the messages thread.
5. Client resubmits the complete messages log back to `/v1/chat/completions`.
6. Gateway returns the final, conversational answer!
