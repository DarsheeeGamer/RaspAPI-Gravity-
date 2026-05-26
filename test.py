import subprocess
import time
import urllib.request
import urllib.error
import json
import sys
import os

GATEWAY_URL = "http://127.0.0.1:8888"

def print_banner(text):
    print("\n" + "=" * 60)
    print(f"🚀 {text}")
    print("=" * 60)

def make_request(url, method="GET", headers=None, data=None):
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    if data is not None:
        req.data = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = response.read().decode("utf-8")
            return response.status, json.loads(res_data)
    except urllib.error.HTTPError as e:
        try:
            res_data = e.read().decode("utf-8")
            return e.code, json.loads(res_data)
        except Exception:
            return e.code, e.reason
    except Exception as e:
        return 0, str(e)

def main():
    print_banner("Starting RASPAPI Integration Tests")
    
    # 1. Start Gateway Server in a Subprocess
    print("⚡ Starting gateway server on port 8888...")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8888"],
        cwd=os.path.join(os.path.dirname(__file__), "src"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the gateway to become active
    connected = False
    for attempt in range(15):
        try:
            status, data = make_request(f"{GATEWAY_URL}/")
            if status == 200:
                print("✅ Gateway server is online and responding!")
                connected = True
                break
        except Exception:
            pass
        time.sleep(1.0)
        
    if not connected:
        print("❌ Failed to connect to the gateway server.")
        process.terminate()
        sys.exit(1)
        
    try:
        # Scenario 1: Generate an API Key
        print_banner("Scenario 1: Generate a fresh API Key")
        status, key_data = make_request(f"{GATEWAY_URL}/api/generate-key", method="POST")
        assert status == 200, f"Failed to generate API key: {key_data}"
        apikey = key_data["apikey"]
        print(f"🔑 Key generated successfully: {apikey}")
        print(f"📋 Status: {key_data.get('status')}, Pool: {key_data.get('pool')}")
        
        # Scenario 2: Test Chat Completions (Default Pool / Fallback using grav_demoapikey)
        print_banner("Scenario 2: Chat completions (Default Pool using grav_demoapikey)")
        headers = {"Authorization": "Bearer grav_demoapikey"}
        completion_payload = {
            "model": "cursor/auto",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a small haiku abt coding."}
            ]
        }
        
        status, comp_data = make_request(
            f"{GATEWAY_URL}/v1/chat/completions",
            method="POST",
            headers=headers,
            data=completion_payload
        )
        assert status == 200, f"Completions request failed: {comp_data}"
        print("💬 Received completions choice response:")
        print(f"🤖 Response:\n{comp_data['choices'][0]['message']['content']}")
        print(f"\n🏁 Finish Reason: {comp_data['choices'][0]['finish_reason']}")
        print(f"📊 Tokens: {comp_data.get('usage')}")
        
        # Scenario 3: Test Tools Schema Support (Client-side AFC Execution Loop)
        print_banner("Scenario 3: Chat completions with Tools Schema (Client-side AFC)")
        messages_history = [
            {"role": "user", "content": "Fetch the weather for New York."}
        ]
        tool_payload = {
            "model": "cursor/auto",
            "messages": messages_history,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }
        
        status, tool_data = make_request(
            f"{GATEWAY_URL}/v1/chat/completions",
            method="POST",
            headers=headers,
            data=tool_payload
        )
        assert status == 200, f"Tool request failed: {tool_data}"
        print("🛠️ Received response with tool description mapping:")
        
        choice = tool_data['choices'][0]
        assistant_message = choice['message']
        print(f"🤖 Response text: {assistant_message.get('content')}")
        print(f"🏁 Finish Reason: {choice['finish_reason']}")
        print(f"⚙️ Tool Calls: {assistant_message.get('tool_calls')}")
        
        # Execute the tool on the client side if the model requested it!
        if choice['finish_reason'] == "tool_calls" and assistant_message.get('tool_calls'):
            tool_call = assistant_message['tool_calls'][0]
            tool_call_id = tool_call['id']
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
            
            print(f"\n🏃 Client-side executing tool '{tool_name}' for location: '{tool_args.get('location')}'...")
            # Simulate the tool execution locally
            tool_result = {"temperature": "72°F", "condition": "Partly Cloudy", "humidity": "60%"}
            print(f"✅ Local tool execution result: {tool_result}")
            
            # Append the assistant's message (containing tool_calls) to the messages history
            messages_history.append(assistant_message)
            
            # Append the tool execution result message
            messages_history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            })
            
            # Request the gateway again with the complete conversation history!
            print("\n🔄 Sending tool result history back to the gateway...")
            followup_payload = {
                "model": "cursor/auto",
                "messages": messages_history,
                "tools": tool_payload["tools"]
            }
            
            status2, followup_data = make_request(
                f"{GATEWAY_URL}/v1/chat/completions",
                method="POST",
                headers=headers,
                data=followup_payload
            )
            assert status2 == 200, f"Followup request failed: {followup_data}"
            
            followup_choice = followup_data['choices'][0]
            print("\n💬 Final response from model after tool execution:")
            print(f"🤖 Response text:\n{followup_choice['message']['content']}")
            print(f"\n🏁 Finish Reason: {followup_choice['finish_reason']}")
        
        # Scenario 4: Global Metrics Verification
        print_banner("Scenario 4: Verify Telemetry Metrics")
        status, metrics = make_request(f"{GATEWAY_URL}/api/metrics")
        assert status == 200, f"Failed to fetch metrics: {metrics}"
        print(f"📈 Total Requests logged: {metrics.get('total_requests')}")
        print(f"🔥 Most popular model: {metrics.get('most_used_model')}")
        print(f"💚 Status: {metrics.get('status')}")
        
        print_banner("🎉 All Integration Tests Passed Successfully!")
        
    finally:
        # Cleanup: Terminate Uvicorn Subprocess
        print("🔌 Stopping gateway server...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("💤 Gateway server stopped cleanly.")
        except Exception:
            process.kill()
            print("💥 Gateway server forced to stop.")

if __name__ == "__main__":
    main()
