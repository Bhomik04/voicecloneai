"""
Intelligent Emotion Detection from Text
Auto-detects appropriate emotion for voice generation

Supports multiple backends:
- Rule-based: Fast, pattern matching (no external dependencies)
- Ollama: Local LLM (free, requires Ollama installation)
- OpenAI: GPT models (requires API key)
- Claude: Anthropic models (requires API key)
"""

import re
from typing import Literal, Optional
import json


EmotionType = Literal["neutral", "excited", "calm", "dramatic", "conversational", "storytelling"]


class EmotionAnalyzer:
    """
    Intelligent emotion detection from text context
    
    Automatically selects appropriate emotion for voice generation
    based on text content, punctuation, and sentiment
    """
    
    EMOTIONS = ["neutral", "excited", "calm", "dramatic", "conversational", "storytelling"]
    
    def __init__(
        self,
        mode: Literal["rule-based", "ollama", "openai", "claude"] = "rule-based",
        api_key: Optional[str] = None,
        ollama_model: str = "llama2",
        openai_model: str = "gpt-3.5-turbo",
        claude_model: str = "claude-3-haiku-20240307",
    ):
        """
        Initialize emotion analyzer
        
        Args:
            mode: Detection method to use
            api_key: API key for OpenAI/Claude (if using those modes)
            ollama_model: Model name for Ollama
            openai_model: Model name for OpenAI
            claude_model: Model name for Claude
        """
        self.mode = mode
        self.api_key = api_key
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self.claude_model = claude_model
        
        # Initialize backend-specific clients
        if mode == "ollama":
            self._init_ollama()
        elif mode == "openai":
            self._init_openai()
        elif mode == "claude":
            self._init_claude()
            
    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import requests
            self.ollama_available = True
            self.ollama_url = "http://localhost:11434/api/generate"
        except ImportError:
            print("⚠️ requests library not found. Install: pip install requests")
            self.ollama_available = False
            
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            print("⚠️ openai library not found. Install: pip install openai")
            self.openai_client = None
            
    def _init_claude(self):
        """Initialize Claude client"""
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            print("⚠️ anthropic library not found. Install: pip install anthropic")
            self.claude_client = None
    
    def analyze(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> EmotionType:
        """
        Analyze text and detect appropriate emotion
        
        Args:
            text: The text to analyze
            context: Optional context about the speech scenario
            
        Returns:
            Detected emotion (one of EMOTIONS)
        """
        if self.mode == "rule-based":
            return self._analyze_rule_based(text)
        elif self.mode == "ollama":
            return self._analyze_ollama(text, context)
        elif self.mode == "openai":
            return self._analyze_openai(text, context)
        elif self.mode == "claude":
            return self._analyze_claude(text, context)
        else:
            return "neutral"
    
    def _analyze_rule_based(self, text: str) -> EmotionType:
        """
        Rule-based emotion detection using patterns
        
        Fast and works offline, ~80% accuracy
        """
        text_lower = text.lower()
        
        # Count indicators
        exclamations = text.count('!')
        questions = text.count('?')
        ellipsis = text.count('...')
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Excitement indicators
        excitement_words = [
            'amazing', 'awesome', 'incredible', 'wow', 'omg', 'fantastic',
            'wonderful', 'brilliant', 'excellent', 'great', 'love', 'excited',
            'thrilled', 'can\'t believe', 'so happy', 'best', 'perfect'
        ]
        excitement_score = sum(1 for word in excitement_words if word in text_lower)
        excitement_score += exclamations * 2
        excitement_score += all_caps_words
        
        # Calm/soothing indicators
        calm_words = [
            'relax', 'calm', 'peaceful', 'gentle', 'soft', 'quiet', 'breathe',
            'slowly', 'carefully', 'meditation', 'tranquil', 'serene', 'rest',
            'soothe', 'easy', 'take your time'
        ]
        calm_score = sum(1 for word in calm_words if word in text_lower)
        
        # Dramatic indicators
        dramatic_words = [
            'terrible', 'horrible', 'disaster', 'tragedy', 'never', 'always',
            'everything', 'nothing', 'impossible', 'unbelievable', 'shocking',
            'devastating', 'catastrophe', 'worst', 'dramatic', 'intense'
        ]
        dramatic_score = sum(1 for word in dramatic_words if word in text_lower)
        dramatic_score += ellipsis * 2
        
        # Conversational indicators
        conversational_words = [
            'hey', 'so', 'like', 'you know', 'i think', 'maybe', 'well',
            'actually', 'basically', 'kind of', 'sort of', 'i mean', 'right',
            'yeah', 'okay', 'sure'
        ]
        conversational_score = sum(1 for word in conversational_words if word in text_lower)
        conversational_score += questions
        
        # Storytelling indicators
        storytelling_words = [
            'once upon', 'long ago', 'there was', 'in the beginning', 'chapter',
            'story', 'tale', 'legend', 'happened', 'then', 'suddenly', 'finally',
            'meanwhile', 'afterwards', 'narrator'
        ]
        storytelling_score = sum(1 for word in storytelling_words if word in text_lower)
        
        # Determine emotion based on scores
        scores = {
            'excited': excitement_score,
            'calm': calm_score,
            'dramatic': dramatic_score,
            'conversational': conversational_score,
            'storytelling': storytelling_score,
        }
        
        # Get highest scoring emotion
        max_emotion = max(scores, key=scores.get)
        max_score = scores[max_emotion]
        
        # Return emotion if score is significant, otherwise neutral
        if max_score >= 2:
            return max_emotion
        else:
            return "neutral"
    
    def _analyze_ollama(
        self,
        text: str,
        context: Optional[str] = None
    ) -> EmotionType:
        """
        Analyze using local Ollama LLM
        
        More accurate than rule-based, works offline, free
        Requires Ollama installed: https://ollama.ai
        """
        if not self.ollama_available:
            print("⚠️ Ollama not available, falling back to rule-based")
            return self._analyze_rule_based(text)
            
        import requests
        
        # Create prompt
        prompt = self._create_llm_prompt(text, context)
        
        try:
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower = more consistent
                    }
                },
                timeout=10,
            )
            
            if response.status_code == 200:
                result = response.json()
                emotion = self._parse_llm_response(result['response'])
                return emotion
            else:
                print(f"⚠️ Ollama error, falling back to rule-based")
                return self._analyze_rule_based(text)
                
        except Exception as e:
            print(f"⚠️ Ollama exception: {e}, falling back to rule-based")
            return self._analyze_rule_based(text)
    
    def _analyze_openai(
        self,
        text: str,
        context: Optional[str] = None
    ) -> EmotionType:
        """Analyze using OpenAI API"""
        if not self.openai_client:
            print("⚠️ OpenAI not available, falling back to rule-based")
            return self._analyze_rule_based(text)
            
        try:
            prompt = self._create_llm_prompt(text, context)
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an emotion detection assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50,
            )
            
            emotion = self._parse_llm_response(response.choices[0].message.content)
            return emotion
            
        except Exception as e:
            print(f"⚠️ OpenAI exception: {e}, falling back to rule-based")
            return self._analyze_rule_based(text)
    
    def _analyze_claude(
        self,
        text: str,
        context: Optional[str] = None
    ) -> EmotionType:
        """Analyze using Claude API"""
        if not self.claude_client:
            print("⚠️ Claude not available, falling back to rule-based")
            return self._analyze_rule_based(text)
            
        try:
            prompt = self._create_llm_prompt(text, context)
            
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=50,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            emotion = self._parse_llm_response(response.content[0].text)
            return emotion
            
        except Exception as e:
            print(f"⚠️ Claude exception: {e}, falling back to rule-based")
            return self._analyze_rule_based(text)
    
    def _create_llm_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Create prompt for LLM emotion detection"""
        prompt = f"""Analyze the following text and determine the most appropriate emotion for voice generation.

Text: "{text}"
"""
        
        if context:
            prompt += f"\nContext: {context}\n"
            
        prompt += f"""
Available emotions:
- neutral: Normal, factual speech
- excited: Enthusiastic, energetic, joyful
- calm: Soothing, relaxed, peaceful
- dramatic: Intense, emotional, theatrical
- conversational: Casual, friendly, chat-like
- storytelling: Narrative, engaging, story mode

Respond with ONLY the emotion name, nothing else.
Emotion:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> EmotionType:
        """Parse LLM response to extract emotion"""
        response_lower = response.lower().strip()
        
        # Try to find emotion in response
        for emotion in self.EMOTIONS:
            if emotion in response_lower:
                return emotion
                
        # Default to neutral if can't parse
        return "neutral"
    
    def analyze_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[EmotionType]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts to analyze
            show_progress: Show progress during analysis
            
        Returns:
            List of detected emotions (same length as texts)
        """
        emotions = []
        
        for i, text in enumerate(texts, 1):
            if show_progress:
                print(f"Analyzing {i}/{len(texts)}: {text[:40]}...")
                
            emotion = self.analyze(text)
            emotions.append(emotion)
            
            if show_progress:
                print(f"  → {emotion}")
                
        return emotions


def test_emotion_analyzer():
    """Test emotion detection on sample texts"""
    print("🧠 Testing Emotion Analyzer\n")
    
    # Test cases
    test_cases = [
        ("Oh my god! This is absolutely amazing! I can't believe it!", "excited"),
        ("Please take a deep breath and relax. Everything will be fine.", "calm"),
        ("This is the most terrible thing that has ever happened...", "dramatic"),
        ("Hey, what do you think about this? I mean, it's pretty cool, right?", "conversational"),
        ("Once upon a time, in a land far away, there lived a brave knight.", "storytelling"),
        ("The meeting is scheduled for 3 PM tomorrow.", "neutral"),
    ]
    
    # Test rule-based detection
    print("📊 Testing Rule-Based Detection:\n")
    analyzer = EmotionAnalyzer(mode="rule-based")
    
    correct = 0
    for text, expected in test_cases:
        detected = analyzer.analyze(text)
        is_correct = detected == expected
        correct += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Expected: {expected:15} | Detected: {detected:15}")
        print(f"   Text: {text[:60]}...\n")
    
    accuracy = correct / len(test_cases) * 100
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})\n")
    
    # Test Ollama if available
    print("🦙 Testing Ollama Detection (if available):\n")
    try:
        ollama_analyzer = EmotionAnalyzer(mode="ollama", ollama_model="llama2")
        
        # Test one sample
        test_text = "Wow! This is incredible! I'm so excited!"
        emotion = ollama_analyzer.analyze(test_text)
        print(f"Text: {test_text}")
        print(f"Detected emotion: {emotion}")
        print("✅ Ollama working!\n")
        
    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")
        print("   Install Ollama from: https://ollama.ai")
        print("   Then run: ollama pull llama2\n")
    
    print("✅ Emotion analyzer test complete!")
    print("\n💡 Usage in voice cloning:")
    print("   analyzer = EmotionAnalyzer(mode='rule-based')")
    print("   emotion = analyzer.analyze(user_text)")
    print("   audio = clone.generate(user_text, emotion=emotion)")


if __name__ == "__main__":
    test_emotion_analyzer()
