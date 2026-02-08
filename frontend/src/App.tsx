import { useEffect, useMemo, useRef, useState } from 'react';
import { TrackCard } from './components/TrackCard';
import { CustomAudioPlayer } from './components/CustomAudioPlayer';

function App() {
  return (
    <div className="min-h-screen bg-music text-slate-100">
      <header className="sticky top-0 z-30 border-b border-white/10 bg-black/60 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-rose-500/20 to-rose-600/10 ring-1 ring-rose-500/30">
              <svg className="h-6 w-6 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">LLM-Conductor</h1>
              <p className="text-xs text-slate-400">AI-powered music composition studio</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full bg-rose-500/10 px-4 py-2 ring-1 ring-rose-500/20">
              <div className="h-2 w-2 animate-pulse rounded-full bg-rose-400 shadow-lg shadow-rose-400/50" />
              <span className="text-xs font-medium text-rose-300">Live Session</span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        <ChatPanel />
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
  has_previous_version?: boolean;  // NEW: Whether _prev files exist
  previous_version_number?: number | null;  // NEW: Previous version number
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
  prompt?: string;  // Optional prompt data from MIDI-LLM calls
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
          if (wsRef.current) {
            wsRef.current.addEventListener('open', handleOpen);
            wsRef.current.addEventListener('error', handleError);
          }
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
          prompt: payload.data?.prompt,  // Extract prompt if available
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
      if (wsRef.current) {
        wsRef.current.addEventListener('open', handleOpen);
        wsRef.current.addEventListener('error', handleError);
      }
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
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.5fr,1fr]">
      <section className="panel">
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">Session Chat</h2>
            <p className="mt-1 text-sm text-slate-400">Direct the Conductor to create and evolve your composition</p>
          </div>
          <div className="hidden items-center gap-2 rounded-full bg-white/5 px-3 py-1.5 text-xs font-mono md:flex">
            <span className="text-slate-400">ID:</span>
            <span className="text-slate-200">{compositionId?.slice(0, 8) ?? '—'}</span>
          </div>
        </div>

      <div className="mt-6 max-h-[48vh] space-y-1.5 overflow-y-auto px-1">
        {messageList.length === 0 && (
          <div className="rounded-3xl bg-white/5 p-8 text-center backdrop-blur-sm">
            <svg className="mx-auto mb-3 h-12 w-12 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <p className="text-sm font-medium text-slate-300">Your session starts here</p>
            <p className="mt-2 text-xs text-slate-500">Try: "Create a calm piano piece in C major"</p>
          </div>
        )}
        {messageList.map((message, index) => {
          // Check if we should show timestamp (first message or time gap > 2 minutes)
          const showTimestamp = index === 0 || (() => {
            if (index > 0) {
              const prevTime = new Date(`1970-01-01 ${messageList[index - 1].timestamp}`);
              const currTime = new Date(`1970-01-01 ${message.timestamp}`);
              return Math.abs(currTime.getTime() - prevTime.getTime()) > 120000; // 2 min
            }
            return false;
          })();

          // Check if message is from same sender as previous
          const sameAsPrevious = index > 0 && messageList[index - 1].role === message.role;

          return (
            <div key={message.id}>
              {showTimestamp && (
                <div className="my-3 text-center text-[10px] text-slate-500">
                  {message.timestamp}
                </div>
              )}
              <div
                className={`message-bubble flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                style={{
                  animationDelay: `${index * 0.03}s`,
                  marginTop: sameAsPrevious ? '2px' : '8px',
                }}
              >
                <div
                  className={`group relative max-w-[75%] px-4 py-2.5 text-[15px] leading-[1.4] ${
                    message.role === 'user'
                      ? 'rounded-[20px] bg-[#007AFF] text-white shadow-sm'
                      : message.role === 'conductor'
                        ? 'rounded-[20px] bg-gradient-to-br from-slate-700/90 to-slate-800/90 text-white shadow-sm'
                        : 'rounded-[18px] bg-slate-800/60 text-slate-300 shadow-sm'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.role !== 'user' && (
                    <div className="mt-1.5 text-[9px] uppercase tracking-wider opacity-50">
                      {message.role === 'conductor' ? 'Conductor' : 'System'}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        {isGenerating && (
          <div className="message-bubble flex justify-start" style={{ marginTop: '8px' }}>
            <div className="flex items-center gap-2 rounded-[20px] bg-slate-700/90 px-4 py-3 shadow-sm">
              <div className="flex gap-1">
                <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '0ms' }} />
                <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '150ms' }} />
                <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-6">
        <div className="relative">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            rows={3}
            placeholder="Describe your musical instruction..."
            disabled={isGenerating}
            className="input-field w-full resize-none pr-24 disabled:opacity-60"
          />
          <button
            type="button"
            onClick={() => void sendMessage()}
            disabled={isGenerating || !input.trim()}
            className="absolute bottom-3 right-3 flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-rose-500 to-rose-600 text-white shadow-lg shadow-rose-500/25 transition-all hover:scale-105 hover:shadow-rose-500/40 active:scale-95 disabled:opacity-40 disabled:hover:scale-100"
            aria-label="Send message"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>
        <div className="mt-3 flex items-start gap-2 text-xs text-slate-400">
          <svg className="mt-0.5 h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
          <div>
            <p>Press <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px]">Enter</kbd> to send, <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px]">Shift+Enter</kbd> for new line</p>
            <p className="mt-1">Try: "Delete track_2" or "Make track_1 more energetic"</p>
          </div>
        </div>
      </div>

      </section>

      <aside className="panel space-y-6">
        <div className="mb-6">
          <h2 className="text-xl font-semibold">Composition</h2>
          <p className="mt-1 text-sm text-slate-400">
            {tracks.length === 0
              ? 'Tracks will appear here'
              : `${tracks.length} track${tracks.length === 1 ? '' : 's'} in session`}
          </p>
        </div>
        <TrackPanelInline
          completion={lastCompletion}
          pendingTracks={pendingTracks}
          pendingMix={pendingMix}
          onQuickAction={sendMessage}
        />
      </aside>
    </div>
  );
}

function DebugFloat() {
  const [entries, setEntries] = useState<DebugEntry[]>([]);
  const [open, setOpen] = useState(false);
  const [expandedEntries, setExpandedEntries] = useState<Set<string>>(new Set());

  useEffect(() => {
    debugBus = (entry) => {
      setEntries((prev) => [...prev.slice(-99), entry]);
    };
    return () => {
      debugBus = null;
    };
  }, []);

  const toggleExpand = (id: string) => {
    setExpandedEntries((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <aside className={`debug-float ${open ? 'open' : ''}`}>
      <button
        className="debug-toggle group"
        onClick={() => setOpen((prev) => !prev)}
        aria-label={open ? 'Hide Debug' : 'Show Debug'}
      >
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
          />
        </svg>
        <span className="ml-2">{open ? 'Hide' : 'Debug'}</span>
      </button>
      <div className="debug-panel">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold">
              <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-rose-500/20 text-rose-400">
                <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
              </span>
              Debug Log
            </h2>
            <p className="mt-1 text-xs text-slate-400">Backend events • {entries.length} entries</p>
          </div>
          <button
            className="rounded-lg bg-white/5 px-3 py-1.5 text-[10px] uppercase tracking-[0.2em] text-slate-400 transition-colors hover:bg-white/10 hover:text-slate-300"
            onClick={() => setEntries([])}
          >
            Clear
          </button>
        </div>
        <div className="space-y-2 text-xs text-slate-300">
          {entries.length === 0 && (
            <div className="rounded-xl border border-dashed border-white/10 bg-white/5 p-6 text-center">
              <svg className="mx-auto mb-2 h-8 w-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-slate-400">No debug messages yet</p>
            </div>
          )}
          {entries.slice().reverse().map((entry) => {
            const isExpanded = expandedEntries.has(entry.id);
            const hasPrompt = entry.prompt !== undefined;

            return (
              <div
                key={entry.id}
                className="rounded-xl border border-white/5 bg-gradient-to-br from-white/5 to-white/[0.02] px-3 py-2 transition-colors hover:border-white/10 hover:bg-white/[0.08]"
                style={{
                  animation: 'slide-up 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div className="font-mono text-[10px] text-slate-500">{entry.timestamp}</div>
                    <div className="h-1 w-1 rounded-full bg-rose-500/50" />
                  </div>
                  {hasPrompt && (
                    <button
                      onClick={() => toggleExpand(entry.id)}
                      className="text-[10px] text-rose-400 hover:text-rose-300"
                    >
                      {isExpanded ? 'Hide' : 'Show'} Prompt
                    </button>
                  )}
                </div>
                <div className="mt-1.5 whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-slate-300">
                  {entry.message}
                </div>
                {hasPrompt && isExpanded && entry.prompt && (
                  <div className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/10 p-2">
                    <div className="mb-1 text-[9px] uppercase tracking-wider text-amber-400">Prompt to MIDI-LLM:</div>
                    <div className="whitespace-pre-wrap break-words font-mono text-[10px] leading-relaxed text-amber-200">
                      {entry.prompt}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
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
            Master Mix
          </h3>
          {pendingMix && (
            <span className="flex items-center gap-2 text-xs text-rose-300">
              <span className="loading-dot" style={{ width: '6px', height: '6px' }} />
              Mixing…
            </span>
          )}
        </div>
        {completion.mix_audio_url ? (
          <div className="mt-3 overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-white/8 to-white/4 p-4 shadow-xl backdrop-blur-sm">
            <CustomAudioPlayer url={completion.mix_audio_url} label="Mix" variant="mix" />
            <div className="mt-3 flex flex-wrap gap-3 text-xs">
              {completion.mix_audio_url && (
                <a
                  href={completion.mix_audio_url}
                  download
                  className="flex items-center gap-1 text-slate-400 underline decoration-slate-600 transition-colors hover:text-rose-300 hover:decoration-rose-500"
                >
                  <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Master MP3
                </a>
              )}
              {completion.mix_midi_url && (
                <a
                  href={completion.mix_midi_url}
                  download
                  className="flex items-center gap-1 text-slate-400 underline decoration-slate-600 transition-colors hover:text-rose-300 hover:decoration-rose-500"
                >
                  <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Master MIDI
                </a>
              )}
            </div>
          </div>
        ) : (
          <div className="mt-3 rounded-2xl border border-dashed border-white/10 bg-white/5 p-6 text-center">
            <svg className="mx-auto mb-2 h-8 w-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <p className="text-sm text-slate-400">Mix will be available after track generation</p>
          </div>
        )}
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-300">
          Tracks ({completion.tracks.length})
        </h3>

        {completion.tracks.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 p-8 text-center">
            <svg className="mx-auto mb-2 h-10 w-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <p className="text-sm text-slate-400">No tracks yet. Start a conversation to create music.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {completion.tracks.map((track) => (
              <TrackCard
                key={`${track.id}-${track.version}`}
                track={track}
                compositionId={completion.composition_id}
                isPending={pendingTracks.has(track.id)}
                onDelete={() => onQuickAction(`Please delete ${track.id}.`)}
                onRegenerate={() =>
                  onQuickAction(`Please regenerate ${track.id} with a different variation.`)
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

