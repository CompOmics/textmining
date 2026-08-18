import os
from openai import OpenAI
from .logging import get_logger

logger = get_logger(__name__)

class LLMClient:
    def __init__(self, config=None):
        config = config or {}
        self.provider = config.get("provider", "openai")  # 'openai' or 'gemini'
        self.base_url = config.get("base_url", "https://api.openai.com/v1/")
        self.model = config.get("model", "gpt-4o-mini-2024-07-18")
        
        # First check for direct api_key in config
        self.api_key = config.get("api_key", "")
        
        # If not found, try environment variable
        if not self.api_key:
            api_key_var = config.get("api_key_env_var", "LLM_API_KEY")
            self.api_key = os.environ.get(api_key_var, "")
            if not self.api_key:
                logger.warning(f"No API key found in config or environment variable {api_key_var}")

        # Initialize appropriate client
        if self.provider == "gemini":
            self._init_gemini_client()
        else:
          
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _init_gemini_client(self):
        """Initialize Google Generative AI client for Gemini."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini client with model: {self.model}")
        except ImportError:
            raise ImportError("google-generativeai package required for Gemini. Install with: pip install google-generativeai")

    def get_completion(self, messages, temperature=0.0):
        if self.provider == "gemini":
            return self._get_gemini_completion(messages, temperature)
        else:
            return self._get_openai_completion(messages, temperature)
    
    def _get_openai_completion(self, messages, temperature=0.0):
        """Get completion from OpenAI-compatible API."""
        # Import seed from reproducibility module if available
        try:
            from .reproducibility import get_seed
            seed = get_seed()
        except ImportError:
            seed = None
        
        # Build request kwargs
        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature,
        }
        
        # Add seed for reproducibility (OpenAI API supports this)
        if seed is not None:
            kwargs["seed"] = seed
        
        chat_completion = self.client.chat.completions.create(**kwargs)
        return chat_completion.choices[0].message.content
    
    def _get_gemini_completion(self, messages, temperature=0.0):
        """Get completion from Google Gemini API."""
        # Convert OpenAI message format to Gemini format
        gemini_messages = []
        system_prompt = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
        
        # Start chat with history
        chat = self.gemini_model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        # Build final prompt with system instruction if present
        final_prompt = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{final_prompt}"
        
        # Generate response
        generation_config = {"temperature": temperature}
        response = chat.send_message(final_prompt, generation_config=generation_config)
        
        return response.text

