# Project Overview
LLM-Conductor: AI-Powered Music Composition Assistant

---

## Purpose

An intelligent music composition system where **GPT-4o acts as a Conductor** that orchestrates **local MIDI-LLM musicians** to create, refine, and arrange music through natural conversation.

Think of it as:
- **ChatGPT for music composition**
- The Conductor (GPT-4o) understands music structure and user intent
- The Musicians (MIDI-LLM via Ollama) generate actual MIDI notes
- Users interact naturally, like talking to a music producer

---

## Core Architecture

```
User ←→ GPT-4o Conductor (understands music + dialogue)
            ↓
      Analyzes MIDI features
            ↓
      Plans musical changes
            ↓
    Ollama MIDI-LLM Musicians
      (generate MIDI tokens)
            ↓
      Multi-track MIDI → Audio
            ↓
    Web Player (solo/mute tracks)
```

---

## Key Concepts

### 1. The Conductor (GPT-4o)
**Role:** Musical director + conversational interface

**Responsibilities:**
- Understand user's musical intent through dialogue
- Analyze generated music via structured MIDI features:
  - Note density, pitch distribution, rhythm patterns
  - Instrument usage, harmonic content
  - Track relationships (melody vs accompaniment)
- Plan refinements: which track to modify, how to adjust
- Maintain conversation context (not full history, just current state)
- Provide musical feedback in natural language

**Does NOT:**
- Generate MIDI tokens directly
- Know audio signal processing
- Store full conversation history

---

### 2. The Musicians (MIDI-LLM)
**Role:** Instrument specialists

**Model:**
- Base: Llama-3.2-1B fine-tuned on MIDI
- Hosted: Local Ollama (localhost:11434)
- Model name: `midi-llm-q6`

**Capabilities:**
- Generate MIDI token sequences from text prompts
- One musician instance per track (Piano, Drums, Bass, etc.)
- Responds to specific musical instructions from Conductor

**Token Format:**
- AMT (Anticipatory Music Transformer) tokenization
- Triples: (arrival_time, duration, instrument_pitch)
- Delivered as: `<|midi_token_XXXXX|>` in Ollama responses

---

### 3. Multi-Track System
**Strategy:** Sequential composition (v1), expand to parallel later

**Workflow:**
1. User: "Create a calm piano melody"
2. Conductor plans: Track 1 = Piano melody
3. Musician 1 generates MIDI
4. User listens, gives feedback: "Add subtle strings"
5. Conductor plans: Track 2 = String pad (complementary to piano)
6. Musician 2 generates MIDI
7. Tracks merge → Audio synthesis
8. User can solo/mute tracks in player

**Track Management:**
- Each track = one MIDI file
- Tracks combine at synthesis stage
- Conductor remembers track roles (melody, harmony, rhythm)

---

## Conversation Flow

```
User: "I want a jazzy piece"
  ↓
Conductor (GPT-4o):
  - Parses intent
  - Plans: "Jazz = swing rhythm + piano + bass + drums"
  - Generates first track instruction: "Swung jazz piano comping in Bb, medium tempo"
  ↓
Musician (MIDI-LLM): [generates piano MIDI tokens]
  ↓
System: MIDI → Audio synthesis
  ↓
Conductor: "I've created a jazz piano foundation. Here's what I composed:
  - 32 bars, Bb major, 120 BPM
  - Swing feel with chord voicings
  - Would you like to add bass, drums, or adjust the piano?"
  ↓
User: "The piano is too busy, simplify it. Then add bass."
  ↓
Conductor:
  - Analyzes piano track features (note_density: 12/sec → too high)
  - Regenerates Track 1 with instruction: "Sparse jazz piano, shell voicings, quarter note rhythm"
  - Plans Track 2: "Walking jazz bass in Bb, complementary to piano"
  ↓
[Loop continues...]
```

---

## Technical Components

### Backend (Python/FastAPI)

**`conductors/gpt_conductor.py`** (NEW)
- OpenAI GPT-4o client
- System prompt: musical knowledge + feature interpretation
- Converts MIDI features → structured text for GPT
- Parses GPT responses → action plans

**`musicians/midi_llm_musician.py`** (REFACTOR from ollama_client)
- One instance per track
- Generates MIDI tokens for specific instrument/role
- Handles Ollama communication

**`core/audio_synthesis.py`** (NEW)
- MIDI → WAV → MP3 via FluidSynth
- Multi-track mixing
- Per-track audio export (for solo/mute)

**`core/track_manager.py`** (NEW)
- Manages multiple MIDI tracks
- Track metadata (instrument, role, BPM, key)
- Merging/splitting tracks

**`core/feature_extractor.py`** (KEEP)
- Same symbolic analysis
- Per-track features + inter-track relationships

**`api/chat.py`** (NEW)
- Chat endpoint: `POST /api/chat`
- WebSocket: `WS /ws/chat`
- Manages conversation state (not full history)

---

### Frontend (React/TypeScript)

**Chat Interface (like ChatGPT)**
- Message bubbles (user + assistant)
- Text input at bottom
- Streaming responses

**Audio Player (embedded in chat)**
- Play/pause/seek
- Track list with solo/mute buttons
- Waveform visualization (optional)
- Download buttons (MIDI + MP3 per track)

**Current Composition Panel**
- Track list with features
- Key, tempo, duration
- Instrument assignments

---

## Data Flow Example

**User:** "Make it more energetic"

**Conductor receives:**
```json
{
  "user_message": "Make it more energetic",
  "current_state": {
    "tracks": [
      {
        "id": "track_1",
        "instrument": "Piano",
        "role": "melody",
        "features": {
          "note_density": 4.2,
          "tempo_estimate": 80,
          "pitch_mean": 60,
          ...
        }
      }
    ]
  }
}
```

**Conductor (GPT-4o) analyzes:**
- Current piano has low note_density (4.2)
- Tempo is slow (80 BPM)
- → Strategy: increase density + faster tempo

**Conductor outputs:**
```json
{
  "response": "I'll increase the energy by speeding up the tempo and adding more rhythmic activity.",
  "actions": [
    {
      "type": "regenerate_track",
      "track_id": "track_1",
      "instruction": "Energetic piano melody, 140 BPM, 16th note patterns, bright articulation",
      "target_features": {
        "note_density": "8-12",
        "tempo": 140
      }
    }
  ]
}
```

**System executes:**
1. Musician regenerates track_1 MIDI
2. Synthesizes new audio
3. Returns to user with playable result

---

## Configuration

**Environment Variables:**
```bash
OPENAI_API_KEY=sk-...           # Required for Conductor
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=midi-llm-q6
SOUNDFONT_PATH=./soundfonts/FluidR3_GM.sf2
```

**`.env` file** (git-ignored):
```
OPENAI_API_KEY=your_key_here
```

---

## Implementation Priorities

### Phase 1: Core Conductor (Current Sprint)
- [x] GPT-4o integration with structured MIDI features
- [x] Single-track conversation flow
- [x] Audio synthesis + playback
- [x] Chat-style UI

### Phase 2: Multi-Track
- [ ] Track manager + per-track musicians
- [ ] Solo/mute player controls
- [ ] Inter-track feature analysis

### Phase 3: Advanced Features
- [ ] Parallel track generation
- [ ] Style transfer between tracks
- [ ] Harmonic analysis
- [ ] Rhythm quantization

---

## Design Principles

**Conductor is the brain:**
- Musical reasoning happens in GPT-4o
- No hardcoded rules for "good music"
- Learns from user feedback in conversation

**Musicians are specialists:**
- MIDI-LLM only generates notes
- Multiple instances for different tracks
- Conductor coordinates them

**Stateless conversation:**
- Only track current composition state
- Don't accumulate full chat history
- Each message contains necessary context

**User-centric:**
- Natural language, no music theory required
- Real-time audio feedback
- Iterative refinement through dialogue

---

## Constraints

All code and comments in English.
User-facing communication in Chinese (via GPT-4o).
