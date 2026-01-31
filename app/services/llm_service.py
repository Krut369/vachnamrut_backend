from groq import Groq
import google.generativeai as genai
import random
from app.core.settings import settings

class LLMService:
    def __init__(self):
        self.groq_keys = settings.GROQ_API_KEYS
        self.gemini_keys = settings.GEMINI_API_KEYS
        
        self.groq_clients = []
        self.gemini_clients = [] # We just store keys for Gemini, as the client is global usually, but we can manage instances if needed.
        
        self._init_clients()
        
    def _init_clients(self):
        # Init Groq
        if self.groq_keys:
            for key in self.groq_keys:
                try:
                    client = Groq(api_key=key, timeout=30.0)
                    self.groq_clients.append(client)
                except Exception as e:
                    print(f"⚠️ Failed to init Groq key {key[:5]}...: {e}")
            print(f"🤖 LLM Service: Loaded {len(self.groq_clients)} Groq clients.")
        
        # Init Gemini (Validating keys roughly)
        if self.gemini_keys:
             self.gemini_clients = self.gemini_keys
             print(f"✨ LLM Service: Loaded {len(self.gemini_clients)} Gemini keys.")

    def get_groq_client(self):
        if not self.groq_clients: return None
        return random.choice(self.groq_clients)

    def get_gemini_key(self):
        if not self.gemini_clients: return None
        return random.choice(self.gemini_clients)

    async def generate_response(self, messages: list, temperature: float = 0.1, json_mode: bool = False, stream: bool = False, provider: str = "groq"):
        """
        Generates response with automatic fallback: Groq -> Gemini -> Fail
        """
        
        # 1. Try Primary Provider (Groq)
        try:
            if provider == "groq" and self.groq_clients:
                return self._call_groq(messages, temperature, json_mode, stream)
        except Exception as e:
            print(f"⚠️ Groq Failed: {e}. Switching to Gemini...")
        
        # 2. Fallback to Gemini
        try:
            if self.gemini_clients:
                print("🔄 Using Gemini Fallback...")
                return self._call_gemini(messages, temperature, json_mode, stream)
        except Exception as e:
            print(f"⚠️ Gemini Failed: {e}")
            
        # 3. Last Resort: Try Groq again if we skipped it initially (e.g. if provider='gemini' failed)
        if provider == "gemini" and self.groq_clients:
             try:
                print("🔄 Switching to Groq...")
                return self._call_groq(messages, temperature, json_mode, stream)
             except Exception as e:
                 print(f"⚠️ Groq Failed: {e}")

        raise Exception("❌ All LLM Providers failed. Please check your API keys or internet connection.")

    def _call_groq(self, messages, temperature, json_mode, stream):
        client = self.get_groq_client()
        kwargs = {
            "model": settings.LLM_MODEL, # "llama-3.3-70b-versatile"
            "messages": messages,
            "temperature": temperature
        }
        if json_mode: kwargs["response_format"] = {"type": "json_object"}
        if stream: kwargs["stream"] = True
        return client.chat.completions.create(**kwargs)

    def _call_gemini(self, messages, temperature, json_mode, stream):
        # Configure the key for this request
        key = self.get_gemini_key()
        genai.configure(api_key=key)
        
        # Convert OpenAI messages to Gemini format
        # System prompt -> system_instruction if possible, or merged into history
        # Gemini 1.5 Pro or Flash
        model_name = "gemini-1.5-flash" 
        
        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            elif msg['role'] == 'user':
                contents.append({"role": "user", "parts": [msg['content']]})
            elif msg['role'] == 'assistant':
                contents.append({"role": "model", "parts": [msg['content']]})

        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain"
        )

        if stream:
            # Gemini stream response
            response = model.generate_content(contents, stream=True, generation_config=generation_config)
            # We need to wrap this in a generator that matches OpenAI style chunks for the Orchestrator
            return self._gemini_stream_wrapper(response)
        
        response = model.generate_content(contents, generation_config=generation_config)
        
        # Mock OpenAI response object for compatibility
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response.text))])

    def _gemini_stream_wrapper(self, response_stream):
        """Yields objects with .choices[0].delta.content to match Groq/OpenAI format"""
        for chunk in response_stream:
             if chunk.text:
                 yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk.text))])

# Helper class for mocking
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

llm_service = LLMService()
