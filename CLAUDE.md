# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLM-Conductor: An AI music composition assistant where **GPT-4o acts as a Conductor** orchestrating **local MIDI-LLM musicians** through natural conversation.

**Not a simple MIDI generator** — this is ChatGPT for music, where:
- Users talk to an AI Conductor (GPT-4o) about musical ideas
- Conductor analyzes MIDI features and plans changes
- MIDI-LLM musicians execute the technical generation
- Multi-track composition with solo/mute playback

## Architecture

```
User ←→ GPT-4o Conductor (musical reasoning)
          ↓
    Analyzes MIDI features (structured data)
          ↓
    Plans actions (generate/modify tracks)
          ↓
    MIDI-LLM Musicians (Ollama, per-track)
          ↓
    Multi-track MIDI → Audio (FluidSynth)
          ↓
    Web Player (solo/mute per track)
```

## Key Technical Details

### GPT-4o Conductor
- **Input**: User message + current composition state (tracks with MIDI features)
- **Output**: Natural language response + structured actions (JSON)
- **System Prompt**: Musical knowledge, feature interpretation, action planning
- **Stateless**: No full chat history, only current state
- **API**: OpenAI API with `gpt-4o` model

### MIDI-LLM Musicians
- **Model**: Llama-3.2-1B fine-tuned for MIDI (via Ollama)
- **Token Format**: `<|midi_token_XXXXX|>` where XXXXX ∈ [0, 55025] (AMT space)
- **Token Cleaning**: Remove all BOS (55026), skip leading non-time tokens
- **One musician per track**: Each generates MIDI for specific instrument/role

### Multi-Track System
- **Track**: MIDI file + metadata (instrument, role, key, tempo, features)
- **Strategy**: Sequential composition (v1) → add tracks one by one
- **Audio**: Per-track MP3 files + mixed master
- **Playback**: Solo/mute controls in web player

### Audio Synthesis
- **Pipeline**: MIDI → WAV (FluidSynth) → MP3 (ffmpeg)
- **Per-track**: Each track synthesized separately for solo/mute
- **Master mix**: All tracks combined
- **SoundFont**: FluidR3_GM.sf2 (General MIDI)

## File Structure

```
backend/app/
├── conductors/
│   └── gpt_conductor.py         # GPT-4o client, system prompt, action parsing
├── musicians/
│   └── midi_llm_musician.py     # Ollama MIDI-LLM per-track generator
├── core/
│   ├── track_manager.py         # Multi-track state management
│   ├── audio_synthesis.py       # MIDI → MP3 conversion
│   ├── feature_extractor.py     # Symbolic music analysis
│   └── token_processor.py       # MIDI token → file conversion
├── api/
│   ├── chat.py                  # POST /api/chat, WS /ws/chat
│   └── routes.py                # Legacy endpoints
└── config.py                    # Settings (requires OPENAI_API_KEY)

frontend/src/
├── components/
│   ├── ChatInterface.tsx        # ChatGPT-style message bubbles
│   ├── AudioPlayer.tsx          # Multi-track player with solo/mute
│   └── CompositionPanel.tsx     # Current track list + features
└── hooks/
    └── useChat.ts               # WebSocket chat state management
```

## Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-...            # REQUIRED for GPT-4o Conductor
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=midi-llm-q6
SOUNDFONT_PATH=./soundfonts/FluidR3_GM.sf2
OUTPUT_DIR=./outputs
```

**IMPORTANT**:
- User must provide their own OpenAI API key in `.env`
- `.env` is git-ignored
- Backend will fail to start without OPENAI_API_KEY

## Commands

### Development
```bash
# Setup
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...

# Start backend (Docker)
docker compose up backend -d

# Start frontend (local)
cd frontend && npm run dev

# Access
http://localhost:5173  # Frontend
http://localhost:8000  # Backend API
```

### Notes
- Keep the default dev workflow above. Use Docker Compose for the backend and local Vite for the frontend.
- The Python virtualenv is only needed for running unit tests locally, not for normal development.

### API Endpoints

**Chat (new system)**
```bash
# POST chat message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a calm piano piece",
    "composition_state": null
  }'

# WebSocket (streaming)
wscat -c ws://localhost:8000/ws/chat
> {"message": "Create a calm piano piece"}
```

**Audio**
```bash
# Get track audio
curl http://localhost:8000/api/tracks/{track_id}/audio -o track.mp3

# Get master mix
curl http://localhost:8000/api/composition/{comp_id}/audio -o mix.mp3
```

### Docker
```bash
docker compose build backend           # After dependency changes
docker compose restart backend         # After code changes (volume mounted)
docker compose logs backend --tail 50  # Debug
```

## Key Implementation Notes

### GPT-4o System Prompt Design
The Conductor's system prompt must:
1. Define its role as musical director (NOT token generator)
2. Explain MIDI feature schema (note_density, pitch_range, etc.)
3. Provide action format (JSON with type + parameters)
4. Emphasize natural Chinese responses to user

### MIDI Feature → GPT Input Format
```python
# Convert MusicFeatures to GPT-friendly text
f"Track 1 (Piano): {features.note_count} notes over {features.duration_seconds}s, "
f"density {features.note_density} notes/sec, pitch range {features.pitch_range[0]}-{features.pitch_range[1]}"
```

### Action Parsing from GPT Response
GPT-4o outputs actions as JSON within its response:
```json
{
  "response": "我来为你创作一段钢琴旋律...",
  "actions": [
    {
      "type": "create_track",
      "instrument": "Piano",
      "role": "melody",
      "instruction": "Calm piano melody in C major, slow tempo, sparse notes"
    }
  ]
}
```

Parse this and execute via musicians.

### Track State Management
```python
@dataclass
class Track:
    id: str
    instrument: str
    role: str  # melody, harmony, rhythm, bass
    midi_path: str
    audio_path: str
    features: MusicFeatures
    metadata: dict  # key, tempo, bars
```

Conductor receives full track list in each message, decides modifications.

## Testing

### Unit Tests
```bash
# Test GPT-4o parsing
pytest backend/app/tests/test_gpt_conductor.py

# Test track management
pytest backend/app/tests/test_track_manager.py

# Test audio synthesis
pytest backend/app/tests/test_audio_synthesis.py
```

### Integration Tests
```bash
# Full chat flow (requires OPENAI_API_KEY)
pytest backend/app/tests/test_chat_integration.py
```

## Common Issues

### "OpenAI API key not configured"
→ Set `OPENAI_API_KEY` in `.env` file

### "anticipation module not found" in Docker
→ Python 3.12 required (not 3.14), already in Dockerfile

### "Ollama connection failed"
→ Run `ollama serve` and `ollama pull midi-llm-q6`

### Audio synthesis fails
→ Check SoundFont exists at `SOUNDFONT_PATH`

## Development Workflow

1. User sends message → `POST /api/chat`
2. Backend constructs GPT prompt with current state
3. GPT-4o responds with actions
4. Actions executed:
   - `create_track` → musician generates MIDI → synthesize audio
   - `regenerate_track` → musician regenerates → synthesize
   - `add_track` → new musician generates → mix with existing
5. Updated state + audio URLs returned to frontend
6. Frontend plays audio, displays tracks
7. User gives feedback → loop continues

## Memory Notes

- **Token cleaning critical**: Remove BOS (55026), skip leading non-time tokens
- **numpy.int64 serialization**: Cast to `int()` before JSON
- **Docker networking**: Use `host.docker.internal:11434` for Ollama
- **Python 3.14 incompatible**: Use 3.12 in Docker for anticipation/librosa
- **Mido vs PrettyMIDI**: `anticipation` returns `mido.MidiFile`, use `pretty_midi` for analysis
