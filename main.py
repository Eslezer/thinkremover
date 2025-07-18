import json
import time
import requests
import traceback
import os
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from urllib.parse import urljoin, urlparse

# Initialize Flask app
app = Flask(__name__)
CORS(app)

def remove_thinking_content(content):
    """
    Remove thinking tags and everything before them from content.
    Returns cleaned content with thinking process removed.
    """
    if not content or not isinstance(content, str):
        return content
    
    # Look for thinking tags
    think_start = content.find('<think>')
    think_end = content.find('</think>')
    
    # If we have both opening and closing think tags
    if think_start != -1 and think_end != -1 and think_start < think_end:
        # Remove everything from start of content up to and including </think>
        cleaned_content = content[think_end + 8:].strip()  # +8 for '</think>'
        return cleaned_content
    
    # If we only have opening think tag, remove everything from that point
    elif think_start != -1:
        # Check if there's content before the think tag
        before_think = content[:think_start].strip()
        if before_think:
            return before_think
        else:
            # If nothing meaningful before think tag, return empty or look for response tags
            response_start = content.find('<response>')
            if response_start != -1:
                return content[response_start:].strip()
            return ""
    
    # If we only have closing think tag, remove everything up to and including it
    elif think_end != -1:
        cleaned_content = content[think_end + 8:].strip()  # +8 for '</think>'
        return cleaned_content
    
    # No thinking tags found, return original content
    return content

def create_error_response(error_message):
    """Create OpenAI-compatible error response"""
    return {
        "choices": [{ 
            "message": { 
                "content": f"Proxy Error: {error_message}" 
            }, 
            "finish_reason": "error" 
        }]
    }

def create_error_stream_chunk(error_message):
    """Create OpenAI-compatible error stream chunk"""
    error_chunk = {
        "choices": [{
            "delta": { "content": f"Proxy Error: {error_message}" },
            "finish_reason": "error"
        }]
    }
    return f'data: {json.dumps(error_chunk)}\n\n'

class ThinkingRemovalParser:
    """Parser for removing thinking content from streaming responses"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.buffer = ""
        self.thinking_started = False
        self.thinking_ended = False
        self.content_to_send = ""
    
    def process_chunk(self, chunk_content):
        """
        Process a chunk and remove thinking content.
        Returns: (content_to_send, is_thinking_complete)
        """
        if not chunk_content:
            return "", False
            
        self.buffer += chunk_content
        
        # If we haven't started thinking detection yet
        if not self.thinking_started and not self.thinking_ended:
            # Check if thinking starts in this chunk
            think_start = self.buffer.find('<think>')
            if think_start != -1:
                # Send any content before the think tag
                content_before = self.buffer[:think_start]
                self.buffer = self.buffer[think_start:]
                self.thinking_started = True
                return content_before, False
            else:
                # No thinking tag found, check for </think> (in case we missed the start)
                think_end = self.buffer.find('</think>')
                if think_end != -1:
                    # Remove everything up to and including </think>
                    self.buffer = self.buffer[think_end + 8:]
                    self.thinking_ended = True
                    content_to_send = self.buffer
                    self.buffer = ""
                    return content_to_send, True
                else:
                    # No thinking tags, send the content as-is
                    content_to_send = self.buffer
                    self.buffer = ""
                    return content_to_send, False
        
        # If we're in thinking mode
        elif self.thinking_started and not self.thinking_ended:
            # Look for the end of thinking
            think_end = self.buffer.find('</think>')
            if think_end != -1:
                # Remove everything up to and including </think>
                self.buffer = self.buffer[think_end + 8:]
                self.thinking_ended = True
                content_to_send = self.buffer
                self.buffer = ""
                return content_to_send, True
            else:
                # Still in thinking mode, don't send anything
                return "", False
        
        # If thinking has ended, send everything
        else:
            content_to_send = self.buffer
            self.buffer = ""
            return content_to_send, True

def get_provider_url():
    """Extract provider URL from request headers or query parameters"""
    # Try header first
    provider_url = request.headers.get('X-Provider-URL')
    
    # Try query parameter
    if not provider_url:
        provider_url = request.args.get('provider_url')
    
    # Try from JSON body
    if not provider_url and request.is_json:
        json_data = request.get_json()
        provider_url = json_data.get('provider_url')
    
    return provider_url

def forward_request_to_provider(provider_url, original_request_data, is_streaming=False):
    """Forward the request to the specified provider"""
    
    # Prepare headers (exclude host and other problematic headers)
    forward_headers = {}
    for key, value in request.headers:
        if key.lower() not in ['host', 'content-length', 'x-provider-url']:
            forward_headers[key] = value
    
    # Make sure we have the right content type
    forward_headers['Content-Type'] = 'application/json'
    
    # Remove provider_url from the request data if it exists
    if isinstance(original_request_data, dict) and 'provider_url' in original_request_data:
        original_request_data = original_request_data.copy()
        del original_request_data['provider_url']
    
    try:
        # Make the request to the provider
        response = requests.post(
            provider_url,
            json=original_request_data,
            headers=forward_headers,
            stream=is_streaming,
            timeout=None  # No timeout as requested
        )
        
        return response
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to provider: {str(e)}")

@app.route('/', methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "Thinking Remover Proxy",
        "version": "1.0.0",
        "description": "Removes <think> tags and thinking content from AI provider responses"
    })

@app.route('/v1/chat/completions', methods=["POST"])
@app.route('/chat/completions', methods=["POST"])
def proxy_chat_completions():
    """Main proxy endpoint for chat completions"""
    
    try:
        # Get the provider URL
        provider_url = get_provider_url()
        
        if not provider_url:
            return jsonify(create_error_response(
                "Provider URL is required. Set it via X-Provider-URL header, provider_url query parameter, or in JSON body."
            )), 400
        
        # Validate provider URL
        try:
            parsed = urlparse(provider_url)
            if not parsed.scheme or not parsed.netloc:
                return jsonify(create_error_response("Invalid provider URL format")), 400
        except Exception:
            return jsonify(create_error_response("Invalid provider URL format")), 400
        
        # Get request data
        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify(create_error_response("Invalid JSON in request body")), 400
        except Exception as e:
            return jsonify(create_error_response(f"Failed to parse JSON: {str(e)}")), 400
        
        is_streaming = request_data.get('stream', False)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Proxying to: {provider_url}")
        print(f"Streaming: {is_streaming}")
        
        # Forward request to provider
        try:
            provider_response = forward_request_to_provider(provider_url, request_data, is_streaming)
        except Exception as e:
            return jsonify(create_error_response(str(e))), 502
        
        # Check if provider returned an error
        if provider_response.status_code != 200:
            try:
                error_data = provider_response.json()
                return jsonify(error_data), provider_response.status_code
            except:
                return jsonify(create_error_response(
                    f"Provider returned status {provider_response.status_code}: {provider_response.text[:200]}"
                )), provider_response.status_code
        
        if is_streaming:
            # Handle streaming response
            def generate_cleaned_stream():
                parser = ThinkingRemovalParser()
                
                try:
                    for line in provider_response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            
                            # Handle server-sent events format
                            if line_str.startswith('data: '):
                                data_str = line_str[6:].strip()
                                
                                if data_str == '[DONE]':
                                    yield 'data: [DONE]\n\n'
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    
                                    # Extract content from the chunk
                                    content_delta = ""
                                    if 'choices' in data and data['choices']:
                                        choice = data['choices'][0]
                                        if 'delta' in choice and 'content' in choice['delta']:
                                            content_delta = choice['delta']['content']
                                    
                                    # Process through thinking remover
                                    if content_delta:
                                        cleaned_content, thinking_complete = parser.process_chunk(content_delta)
                                        
                                        # If we have content to send, modify the chunk
                                        if cleaned_content:
                                            # Update the chunk with cleaned content
                                            if 'choices' in data and data['choices']:
                                                data['choices'][0]['delta']['content'] = cleaned_content
                                            yield f'data: {json.dumps(data)}\n\n'
                                    else:
                                        # No content delta, pass through as-is
                                        yield f'data: {json.dumps(data)}\n\n'
                                        
                                except json.JSONDecodeError:
                                    # If we can't parse JSON, pass through as-is
                                    yield line_str + '\n'
                            else:
                                # Pass through non-data lines
                                yield line_str + '\n'
                                
                except Exception as e:
                    print(f"Error in streaming: {e}")
                    yield create_error_stream_chunk(f"Streaming error: {str(e)}")
                    yield 'data: [DONE]\n\n'
                finally:
                    provider_response.close()
            
            return Response(
                stream_with_context(generate_cleaned_stream()),
                content_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        else:
            # Handle non-streaming response
            try:
                response_data = provider_response.json()
                
                # Process each choice to remove thinking content
                if 'choices' in response_data:
                    for choice in response_data['choices']:
                        if 'message' in choice and 'content' in choice['message']:
                            original_content = choice['message']['content']
                            cleaned_content = remove_thinking_content(original_content)
                            choice['message']['content'] = cleaned_content
                
                return jsonify(response_data)
                
            except json.JSONDecodeError:
                return jsonify(create_error_response("Provider returned invalid JSON")), 502
            except Exception as e:
                return jsonify(create_error_response(f"Error processing response: {str(e)}")), 500
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return jsonify(create_error_response(f"Internal server error: {str(e)}")), 500

# Catch-all route for other endpoints
@app.route('/<path:path>', methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy_other_endpoints(path):
    """Proxy other endpoints to the provider"""
    
    try:
        provider_url = get_provider_url()
        
        if not provider_url:
            return jsonify(create_error_response(
                "Provider URL is required for proxying"
            )), 400
        
        # Build the full URL
        full_url = urljoin(provider_url.rstrip('/') + '/', path)
        
        # Forward headers
        forward_headers = {}
        for key, value in request.headers:
            if key.lower() not in ['host', 'content-length', 'x-provider-url']:
                forward_headers[key] = value
        
        # Forward the request
        if request.method == 'GET':
            response = requests.get(
                full_url,
                headers=forward_headers,
                params=request.args,
                timeout=None
            )
        else:
            response = requests.request(
                method=request.method,
                url=full_url,
                headers=forward_headers,
                json=request.get_json() if request.is_json else None,
                data=request.get_data() if not request.is_json else None,
                params=request.args,
                timeout=None
            )
        
        # Return the response
        try:
            return jsonify(response.json()), response.status_code
        except:
            return response.text, response.status_code, {'Content-Type': response.headers.get('Content-Type', 'text/plain')}
    
    except Exception as e:
        return jsonify(create_error_response(f"Proxy error: {str(e)}")), 502

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" Thinking Remover Proxy Server")
    print(" Removes <think> tags from AI provider responses")
    print(" Usage: Set provider URL via X-Provider-URL header or provider_url parameter")
    print(" Example: your-server.com/v1/chat/completions?provider_url=https://api.magistral.ai/v1/chat/completions")
    print("=" * 60 + "\n")
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
