import { useEffect, useMemo, useRef, useState } from 'react';

function App() {
  return (
    <div className="min-h-screen bg-music text-slate-100">
      <header className="border-b border-white/10 bg-black/40 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">LLM-Conductor</h1>
            <p className="text-sm text-slate-300">Conversational music composition studio</p>
          </div>
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Live Session</div>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 px-6 py-8 lg:grid-cols-[1.6fr,1fr]">
        <ChatPanel />
        <div className="hidden lg:block" aria-hidden="true" />
      </main>

      <DebugFloat />
    </div>
  );
}

export default App;

type ChatMessage = {
  id: string;
  role: 'user' | 'conductor' | 'system';
  content: string;
  timestamp: string;
};

type TrackInfo = {
  id: string;
  instrument: string;
  role: string;
  midi_url: string;
  audio_url: string;
  features: Record<string, unknown>;
  version: number;
};

type CompletionPayload = {
  composition_id: string;
  tracks: TrackInfo[];
  mix_midi_url?: string | null;
  mix_audio_url?: string | null;
};

type DebugEntry = {
  id: string;
  message: string;
  timestamp: string;
};

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/ws/chat`;

let debugBus: ((entry: DebugEntry) => void) | null = null;

function ChatPanel() {
  const [compositionId, setCompositionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastCompletion, setLastCompletion] = useState<CompletionPayload | null>(null);
  const [pendingTracks, setPendingTracks] = useState<Set<string>>(new Set());
  const [pendingMix, setPendingMix] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = (): Promise<WebSocket> => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        return Promise.resolve(wsRef.current);
      }
      if (wsRef.current.readyState === WebSocket.CONNECTING) {
        return new Promise((resolve, reject) => {
          const handleOpen = () => {
            wsRef.current?.removeEventListener('open', handleOpen);
            wsRef.current?.removeEventListener('error', handleError);
            if (wsRef.current) resolve(wsRef.current);
          };
          const handleError = () => {
            wsRef.current?.removeEventListener('open', handleOpen);
            wsRef.current?.removeEventListener('error', handleError);
            reject(new Error('WebSocket connection failed'));
          };
          wsRef.current.addEventListener('open', handleOpen);
          wsRef.current.addEventListener('error', handleError);
        });
      }
    }

    wsRef.current = new WebSocket(WS_URL);
    wsRef.current.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      const type = payload.type;
      if (type === 'debug') {
        debugBus?.({
          id: crypto.randomUUID(),
          message: payload.data?.message || JSON.stringify(payload.data),
          timestamp: new Date().toLocaleTimeString(),
        });
      }
      if (type === 'status') {
        appendMessage({
          role: 'system',
          content: payload.data?.message || 'Working...',
        });
      }
      if (type === 'action') {
        const actionType = payload.data?.action_type;
        const params = payload.data?.parameters || {};
        if (actionType === 'create_track') {
          setPendingMix(true);
          appendMessage({
            role: 'system',
            content: 'Creating a new track…',
          });
        }
        if (actionType === 'regenerate_track' && params.track_id) {
          setPendingTracks((prev) => new Set(prev).add(params.track_id));
          setPendingMix(true);
          appendMessage({
            role: 'system',
            content: `Updating ${params.track_id}…`,
          });
        }
        if (actionType === 'delete_track' && params.track_id) {
          setPendingMix(true);
          appendMessage({
            role: 'system',
            content: `Deleting ${params.track_id}…`,
          });
        }
      }
      if (type === 'track_generated') {
        appendMessage({
          role: 'system',
          content: `Track ${payload.data?.track_id} is ready.`,
        });
      }
      if (type === 'track_updated') {
        appendMessage({
          role: 'system',
          content: `Track ${payload.data?.track_id} has been refreshed.`,
        });
      }
      if (type === 'conductor_message') {
        appendMessage({
          role: 'conductor',
          content: payload.data?.message || '',
        });
      }
      if (type === 'completed') {
        const completion: CompletionPayload = payload.data;
        setCompositionId(completion.composition_id);
        setLastCompletion(completion);
        setIsGenerating(false);
        setPendingTracks(new Set());
        setPendingMix(false);
      }
      if (type === 'error') {
        appendMessage({
          role: 'system',
          content: payload.data?.message || 'An error occurred.',
        });
        debugBus?.({
          id: crypto.randomUUID(),
          message: payload.data?.message || 'Backend error',
          timestamp: new Date().toLocaleTimeString(),
        });
        setIsGenerating(false);
        setPendingTracks(new Set());
        setPendingMix(false);
      }
    };
    wsRef.current.onclose = () => {
      wsRef.current = null;
    };

    return new Promise((resolve, reject) => {
      const handleOpen = () => {
        wsRef.current?.removeEventListener('open', handleOpen);
        wsRef.current?.removeEventListener('error', handleError);
        if (wsRef.current) resolve(wsRef.current);
      };
      const handleError = () => {
        wsRef.current?.removeEventListener('open', handleOpen);
        wsRef.current?.removeEventListener('error', handleError);
        reject(new Error('WebSocket connection failed'));
      };
      wsRef.current.addEventListener('open', handleOpen);
      wsRef.current.addEventListener('error', handleError);
    });
  };

  const appendMessage = (partial: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    const id = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id, timestamp: new Date().toLocaleTimeString(), ...partial },
    ]);
  };

  const sendMessage = async (content?: string) => {
    const payload = (content ?? input).trim();
    if (!payload) return;
    appendMessage({ role: 'user', content: payload });
    setIsGenerating(true);
    try {
      const ws = await connect();
      ws.send(
        JSON.stringify({
          composition_id: compositionId,
          message: payload,
        }),
      );
    } catch (error) {
      appendMessage({
        role: 'system',
        content: error instanceof Error ? error.message : 'WebSocket error.',
      });
      setIsGenerating(false);
    }
    setInput('');
  };

  const messageList = useMemo(() => messages, [messages]);
  const tracks = lastCompletion?.tracks ?? [];

  return (
    <section className="panel">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold">Session Chat</h2>
          <p className="text-sm text-slate-400">Give the Conductor instructions and evolve the arrangement.</p>
        </div>
        <div className="hidden items-center gap-2 text-xs text-slate-400 md:flex">
          <span>{tracks.length} tracks</span>
          <span className="h-1 w-1 rounded-full bg-rose-500" />
          <span>{compositionId ?? 'Not started'}</span>
        </div>
      </div>

      <div className="mt-6 max-h-[48vh] space-y-4 overflow-y-auto">
        {messageList.length === 0 && (
          <div className="skeleton-block">Your session starts here. Describe a mood or instrument.</div>
        )}
        {messageList.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-lg ${
                message.role === 'user'
                  ? 'bg-white text-slate-900'
                  : message.role === 'conductor'
                    ? 'bg-rose-500/15 text-rose-50'
                    : 'bg-white/5 text-slate-200'
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              <div className="mt-2 text-[11px] uppercase tracking-widest opacity-60">
                {message.role}
              </div>
            </div>
          </div>
        ))}
        {isGenerating && (
          <div className="loading-callout">
            <span className="loading-dot" />
            Conductor is working…
          </div>
        )}
      </div>

      <div className="mt-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            rows={2}
            placeholder="Describe your next musical instruction..."
            className="input-field"
          />
          <button
            type="button"
            onClick={() => void sendMessage()}
            disabled={isGenerating}
            className="btn-primary"
          >
            Send
          </button>
        </div>
        <div className="mt-3 text-xs text-slate-400">
          Tip: “Delete track_2” or “Regenerate track_1 with softer strings.”
        </div>
      </div>

      <div className="mt-6 border-t border-white/5 pt-6">
        <TrackPanelInline
          completion={lastCompletion}
          pendingTracks={pendingTracks}
          pendingMix={pendingMix}
          onQuickAction={sendMessage}
        />
      </div>
    </section>
  );
}

function DebugFloat() {
  const [entries, setEntries] = useState<DebugEntry[]>([]);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    debugBus = (entry) => {
      setEntries((prev) => [...prev.slice(-199), entry]);
    };
    return () => {
      debugBus = null;
    };
  }, []);

  return (
    <aside className={`debug-float ${open ? 'open' : ''}`}>
      <button className="debug-toggle" onClick={() => setOpen((prev) => !prev)}>
        {open ? 'Hide Debug' : 'Show Debug'}
      </button>
      <div className="debug-panel">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-sm font-semibold">Debug Log</h2>
            <p className="text-xs text-slate-400">Live backend traces.</p>
          </div>
          <button
            className="text-xs uppercase tracking-[0.3em] text-slate-400"
            onClick={() => setEntries([])}
          >
            Clear
          </button>
        </div>
        <div className="mt-4 space-y-3 text-xs text-slate-300">
          {entries.length === 0 && (
            <div className="skeleton-block">No debug messages yet.</div>
          )}
          {entries.map((entry) => (
            <div key={entry.id} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <div className="text-[11px] uppercase tracking-widest text-slate-500">
                {entry.timestamp}
              </div>
              <div className="mt-2 whitespace-pre-wrap text-slate-200">{entry.message}</div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function TrackPanelInline({
  completion,
  pendingTracks,
  pendingMix,
  onQuickAction,
}: {
  completion: CompletionPayload | null;
  pendingTracks: Set<string>;
  pendingMix: boolean;
  onQuickAction: (message: string) => void;
}) {
  if (!completion) {
    return <div className="skeleton-block">No tracks yet.</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-300">
            Composition Mix
          </h3>
          {pendingMix && <span className="text-xs text-rose-300">Updating mix…</span>}
        </div>
        {completion.mix_audio_url ? (
          <div className="mt-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
            <AudioPlayer url={completion.mix_audio_url} label="Mix" />
            <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-300">
              {completion.mix_audio_url && (
                <a className="underline" href={completion.mix_audio_url} download>
                  Download MP3
                </a>
              )}
              {completion.mix_midi_url && (
                <a className="underline" href={completion.mix_midi_url} download>
                  Download MIDI
                </a>
              )}
            </div>
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-400">Mix not available yet.</p>
        )}
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-300">Tracks</h3>
        {completion.tracks.map((track) => (
          <div
            key={`${track.id}-${track.version}`}
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4 transition hover:-translate-y-0.5 hover:bg-white/10"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-white">{track.instrument}</p>
                <p className="text-xs text-slate-400">{track.role} • {track.id}</p>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-300">
                <button
                  className="btn-secondary"
                  onClick={() => onQuickAction(`Please delete ${track.id}.`)}
                >
                  Delete
                </button>
                <a className="underline" href={track.audio_url} download>
                  MP3
                </a>
                <a className="underline" href={track.midi_url} download>
                  MIDI
                </a>
              </div>
            </div>
            <div className="mt-3">
              {pendingTracks.has(track.id) && (
                <div className="mb-2 text-xs text-rose-300">Refreshing track…</div>
              )}
              <AudioPlayer url={track.audio_url} label={track.id} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AudioPlayer({ url, label }: { url: string; label: string }) {
  const [loading, setLoading] = useState(true);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.load();
    }
    setLoading(true);
  }, [url]);

  return (
    <div className="space-y-2">
      {loading && (
        <div className="loading-mini">
          <span className="loading-dot" />
          Loading {label}…
        </div>
      )}
      <audio
        ref={audioRef}
        key={url}
        controls
        className="w-full"
        onLoadStart={() => setLoading(true)}
        onLoadedData={() => setLoading(false)}
        onCanPlay={() => setLoading(false)}
      >
        <source src={url} />
      </audio>
    </div>
  );
}
