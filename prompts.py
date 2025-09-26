"""
Claude Prompts Configuration
Centralized location for all Claude prompts with template variable substitution
"""

from string import Template
from typing import Dict, Any

class PromptTemplates:
    """Centralized prompt templates with variable substitution"""
    
    # Generic building blocks for reuse across prompts
    DELETION_ANALYSIS_BLOCK = """**DELETION ANALYSIS** - You are a thoughtful curator helping someone maintain a meaningful personal voice note archive. 

Your job is to evaluate voice note transcripts and recommend whether they should be KEPT or DELETED based on their potential future value.

DELETE a voice note if it contains:

1. Pure logistics: Temporary reminders that are no longer relevant ("pick up milk," "meeting at 3pm")
2. Technical testing: "Testing 1, 2, 3" or checking if the recording works
3. Incomplete thoughts: Fragment recordings with no meaningful content or context
4. Mundane status updates: Routine daily activities with no emotional or intellectual depth
5. Superseded information: Details that have been updated or replaced by newer information
6. Purely transactional: Basic coordination that served its immediate purpose ("running 5 minutes late")
7. Background noise/accidents: Recordings with no intentional content
8. Repetitive content: Ideas or thoughts already captured more fully in other notes

**For creative/musical content:** KEEP all music/creative recordings unless they are clearly test recordings or accidental captures. Creative expression is valuable and should be preserved.

When in doubt, lean toward KEEPING - future you might find unexpected value in what seems mundane today. Consider the voice note's potential for:

- Triggering forgotten memories
- Revealing patterns in thinking/behavior
- Providing emotional comfort during difficult times
- Inspiring future creative work"""
    
    TAGS_INSTRUCTION_BLOCK = """**TAGS** - Create up to 20 relevant tags (1-3 words each, Title Case, no slashes) that best describe this voice memo. Choose the most important and descriptive tags covering themes, focus areas, content types, emotional tones, and key topics. Ensure that all people, places, companies, location, and other important details are included as tags."""
    
    KEYWORDS_INSTRUCTION_BLOCK = """**KEYWORDS** - Extract a concise list of the most important named entities and symbolic figures that someone might search for later. Extract only: People's names, Organizations/companies/institutions, Cities or specific places, Symbolic figures/metaphors/archetypes (e.g., "lion," "chief," "fire," "mother wound"), Named events or rituals (e.g., "wedding," "Solstice," "interview," "brothers' circle"). Do not include: Emotional states, common actions, relationship terms, or general experiences. Output a flat comma-separated list (Up to 8 terms max)."""
    
    SUMMARY_INSTRUCTION_BLOCK = """**SUMMARY** - Create a concise 2-3 sentence summary that a human could read in 7s or less to understand the content (content_type, content_purpose, content_tone, audience, and key_insights). Write it in short hand so that's it's easy to read quickly."""
    
    STANDARD_OUTPUT_FORMAT = """**FORMAT YOUR RESPONSE EXACTLY AS:**

TITLE: [compelling title here]

PROCESSED_CONTENT:
[appropriate content processing]

SUMMARY: [2-3 sentence summary here]

TAGS: [tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8]

KEYWORDS: [keyword1, keyword2, keyword3, keyword4, keyword5]

DELETION_FLAG: [KEEP/DELETE]
DELETION_REASON: [brief explanation of your reasoning]"""
    
    # Music/Creative Content Processing Prompt
    MUSIC_ANALYSIS_PROMPT = Template(f"""Analyze this audio file that has been classified as music/creative content. Please provide ALL of the following in your response:

**ORIGINAL TRANSCRIPT:**
"${{transcript}}"

**FILENAME:** ${{filename}}

**AUDIO CLASSIFICATION:**
- Content Type: ${{audio_type}}
- Confidence: ${{confidence}}
- Top Predictions: ${{top_predictions}}

**IMPORTANT:** This audio was classified as "${{audio_type}}" with ${{confidence}} confidence. The transcript above may be garbled, repetitive, or nonsensical because this is a musical recording without clear vocals.

Please analyze this creative content and provide:

1. **TITLE** - Create a compelling title based on the filename and any discernible content. If the filename has meaningful information, use and improve it. Use proper Title Case capitalization.

2. **PROCESSED CONTENT** - Since this is a music file:
   - If the transcript is garbled/repetitive (like "24 24 24" or nonsensical patterns): Replace with "No transcript available - this is a music file without clear vocals"
   - If the transcript has some recognizable lyrics/words: Provide those lyrics cleaned up, noting "Partial lyrics from music file:"
   - If completely unclear: "No transcript available - this is an instrumental music file"

3. {SUMMARY_INSTRUCTION_BLOCK}

4. {TAGS_INSTRUCTION_BLOCK}

5. {KEYWORDS_INSTRUCTION_BLOCK}

6. {DELETION_ANALYSIS_BLOCK}

{STANDARD_OUTPUT_FORMAT}""")

    # Speech/Voice Memo Processing Prompt  
    SPEECH_ANALYSIS_PROMPT = Template(f"""Analyze this voice memo transcript and provide a comprehensive analysis. Please provide ALL of the following in your response:

**ORIGINAL TRANSCRIPT:**
"${{transcript}}"

**FILENAME:** ${{filename}}

**AUDIO CLASSIFICATION:**
- Content Type: ${{audio_type}}
- Confidence: ${{confidence}}
- Top Predictions: ${{top_predictions}}

This audio was classified as "${{audio_type}}" with ${{confidence}} confidence, indicating clear speech content.

Please analyze this voice memo and provide:

1. **TITLE** - Create a specific, compelling title (3-8 words) that captures the essence of the content. If the filename contains meaningful information, incorporate and improve it. Use proper Title Case capitalization.

2. **FORMATTED TRANSCRIPT** - Take the transcript and improve readability by fixing grammar, typos, and format for easy human readability (appropriate punctuation, capitalization, and paragraph breaks).  

3. {SUMMARY_INSTRUCTION_BLOCK}

4. {TAGS_INSTRUCTION_BLOCK}

5. {KEYWORDS_INSTRUCTION_BLOCK}

6. {DELETION_ANALYSIS_BLOCK}

**IMPORTANT:** ALWAYS provide the processed content regardless of deletion flag. The deletion analysis is separate from content processing.

{STANDARD_OUTPUT_FORMAT}""")

    # Fallback prompt for unknown/unclassified audio
    UNKNOWN_ANALYSIS_PROMPT = Template(f"""Analyze this voice memo transcript. The audio classification was uncertain.

**ORIGINAL TRANSCRIPT:**
"${{transcript}}"

**FILENAME:** ${{filename}}

**AUDIO CLASSIFICATION:**
- Content Type: ${{audio_type}}
- Confidence: ${{confidence}}

Please analyze this content and provide:

1. **TITLE** - Create a compelling title based on the content and filename. Use proper Title Case capitalization.

2. **PROCESSED_CONTENT** - Improve readability by fixing grammar, typos, and formatting while preserving ALL original ideas, words, and meaning. If the content appears to be garbled music, note that appropriately.

3. {SUMMARY_INSTRUCTION_BLOCK}

4. {TAGS_INSTRUCTION_BLOCK}

5. {KEYWORDS_INSTRUCTION_BLOCK}

6. {DELETION_ANALYSIS_BLOCK}

{STANDARD_OUTPUT_FORMAT}""")

    # Transcript Formatting Prompt - for formatting-only operations
    TRANSCRIPT_FORMAT_PROMPT = Template("""Format this transcript for better readability while preserving ALL original content.

**ORIGINAL TRANSCRIPT:**
"${transcript}"

**FILENAME:** ${filename}

**FORMATTING INSTRUCTIONS:**
You are a thoughtful, voice-centered editor. Transform this raw transcript into clear, emotionally resonant writing that feels like someone thinking out loud, supported by light structure to aid readability.

Guidelines:
1. Write with a natural, expressive voice that matches the speaker's tone and emotional arc
2. Use short to medium-length paragraphs (3–6 lines) with steady, flowing rhythm
3. Add ## section headings for major conceptual or emotional shifts (2–5 headings max)
4. Use **bold** sparingly for phrases with real emotional or thematic weight
5. Use > for deep, reflective thoughts (once or twice maximum)
6. Preserve ALL original ideas, words, and meaning - do not add, remove, or change concepts
7. Keep headings simple and clear - no clever titles

**RESPOND WITH VALID JSON:**
{
  "formatted_transcript": "Your formatted markdown text here"
}""")

    # Semantic Fingerprinting Prompt - for extracting cognitive DNA from insights
    SEMANTIC_FINGERPRINTING_PROMPT = Template("""You are an expert at analyzing insights and extracting their essential cognitive DNA. Your task is to read text document and extract its semantic fingerprint according to the schema below.

## Input

A text document (may contain filler words, incomplete sentences, stream-of-consciousness thinking)

## Output Format

Return a JSON object with the following structure and fields:

```python
{
  "core_exploration": {
    "central_question": "The fundamental inquiry driving this insight",
    "key_tension": "The productive contradiction being reconciled", 
    "breakthrough_moment": "The specific realization that shifted understanding",
    "edge_of_understanding": "What they recognize they haven't figured out yet"
  },
  "conceptual_dna": [
    "2-4 essential concept-patterns that capture core wisdom",
    "Should be quotable standalone insights"
  ],
  "pattern_signature": {
    "thinking_style": ["array of thinking styles from taxonomy"],
    "insight_type": "category from taxonomy",
    "development_stage": "maturity level from taxonomy",
    "confidence_level": 0.0-1.0
  },
  "bridge_potential": {
    "domains_connected": ["array of domains from taxonomy"],
    "novel_synthesis": "unique combination being created",
    "cross_domain_pattern": "universal principle that transcends domains"
  },
  "genius_indicators": {
    "uniqueness_score": 0.0-1.0,
    "depth_score": 0.0-1.0,
    "generative_potential": 0.0-1.0,
    "framework_emergence": 0.0-1.0
  },
  "raw_essence": "2-3 sentences capturing core insight in natural language",
  "embedding_text": "concentrated keywords optimized for similarity search"
}
```

## Taxonomies to Use

**Thinking Styles** (can select multiple):
analytical-linear, systems-holistic, embodied-somatic, narrative-temporal, metaphorical-associative, dialectical-synthetic, intuitive-emergent, experimental-iterative

**Insight Types** (select one):
observation, methodology, framework, philosophy, synthesis, theory, question, distinction

**Development Stages** (select one):
noticing, exploring, developing, breakthrough, integrating, refining, applying, teaching

**Domains** (select relevant ones):
personal_practice, relationships, health_wellness, spirituality, business_strategy, leadership, organizational, career, artistic, design, innovation, systems_thinking, psychology, philosophy, science, community, education, social_change, culture

## Detailed Scoring Instructions

**Confidence Level (0.0-1.0)**
Start with 0.5 baseline, then adjust:

- Add 0.3 for expressions of strong certainty and conviction in the insight
- Add 0.2 for use of definitive, absolute language about the discovery
- Subtract 0.3 for expressions of uncertainty, doubt, or tentative exploration
- Subtract 0.2 for hesitation or acknowledgment of unclear understanding
Final score = clamp between 0.0-1.0

**Uniqueness Score (0.0-1.0)**
Start with 0.0, then add:

- +0.4 if challenges or contradicts widely accepted conventional wisdom
- +0.3 if bridges domains that are rarely connected in novel ways
- +0.3 if introduces genuinely original methodology, metaphor, or perspective
Maximum 1.0

**Depth Score (0.0-1.0)**
Base score by level of understanding demonstrated:

- 0.2 - Surface observation of phenomena
- 0.4 - Recognition of patterns and relationships
- 0.6 - Understanding of underlying mechanisms and causation
- 0.8 - Articulation of fundamental principles
- 1.0 - Recognition of meta-principles and recursive patterns

Additional depth indicators:

- Add 0.1 if explains underlying causation rather than just correlation
- Add 0.1 if addresses root causes rather than surface symptoms
- Add 0.1 if recognizes self-referential or recursive dynamics
Maximum 1.0

**Generative Potential (0.0-1.0)**
Start with 0.0, then add:

- +0.3 if opens new avenues of inquiry (poses questions that weren't obvious before, identifies unexplored possibilities)
- +0.3 if suggests actionable exploration (proposes ways to test, validate, or further develop the insight)
- +0.4 if introduces genuinely novel perspective or framework that could spawn new thinking
Maximum 1.0

**Framework Emergence (0.0-1.0)**
Start with 0.0, then add:

- +0.3 if demonstrates systematic organization (coherent structure that others could follow)
- +0.3 if articulates replicable process (describes methodology that could be applied elsewhere)
- +0.4 if reaches teachable clarity (insight is developed enough that someone could learn and apply it)
Maximum 1.0

## Analysis Guidelines

- Look beneath surface content - What deeper question is really being explored?
- Identify productive tensions - What paradox or contradiction drives creative energy?
- Find the breakthrough - What specific realization shifted understanding?
- Recognize thinking patterns - How is this person processing information?
- Assess domain bridging - What previously unconnected areas are being linked?
- Extract quotable wisdom - What could stand alone as valuable insight?

Read the transcript carefully and extract its semantic fingerprint following this schema exactly.

**Text Document:**
${insight}""")

    @classmethod
    def get_prompt_for_audio_type(cls, audio_type: str, **kwargs) -> str:
        """
        Get the appropriate prompt template for the audio type and substitute variables
        
        Args:
            audio_type: The classified audio type (Music, Speech, etc.)
            **kwargs: Variables to substitute in the template
            
        Returns:
            Fully substituted prompt string
        """
        # Normalize audio type
        audio_type_lower = audio_type.lower() if audio_type else 'unknown'
        
        # Select appropriate template
        if any(music_term in audio_type_lower for music_term in ['music', 'singing', 'song', 'instrumental']):
            template = cls.MUSIC_ANALYSIS_PROMPT
        elif any(speech_term in audio_type_lower for speech_term in ['speech', 'narration', 'monologue', 'conversation']):
            template = cls.SPEECH_ANALYSIS_PROMPT
        else:
            template = cls.UNKNOWN_ANALYSIS_PROMPT
        
        # Ensure all required variables have defaults
        substitution_vars = {
            'transcript': kwargs.get('transcript', ''),
            'filename': kwargs.get('filename', 'Voice Memo'),
            'audio_type': kwargs.get('audio_type', 'Unknown'),
            'confidence': f"{kwargs.get('confidence', 0.0):.3f}",
            'top_predictions': ', '.join([pred[0] for pred in kwargs.get('top_yamnet_predictions', [])[:3]]) or 'None available'
        }
        
        # Substitute variables in template
        try:
            return template.substitute(**substitution_vars)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")

def get_analysis_prompt(audio_type: str, transcript: str, filename: str = '', 
                       audio_classification: Dict[str, Any] = None) -> str:
    """
    Convenience function to get a fully substituted analysis prompt
    
    Args:
        audio_type: The classified audio type
        transcript: The audio transcript
        filename: Original filename
        audio_classification: Full audio classification results
        
    Returns:
        Ready-to-use prompt string
    """
    classification_data = audio_classification or {}
    
    return PromptTemplates.get_prompt_for_audio_type(
        audio_type=audio_type,
        transcript=transcript,
        filename=filename,
        confidence=classification_data.get('confidence', 0.0),
        top_yamnet_predictions=classification_data.get('top_yamnet_predictions', [])
    )