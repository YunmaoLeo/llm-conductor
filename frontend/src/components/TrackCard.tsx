import { useState } from 'react';
import { CustomAudioPlayer } from './CustomAudioPlayer';

interface TrackCardProps {
  track: {
    id: string;
    instrument: string;
    role: string;
    midi_url: string;
    audio_url: string;
    features: Record<string, unknown>;
    version: number;
  };
  isPending: boolean;
  onDelete: () => void;
  onRegenerate: () => void;
}

export function TrackCard({
  track,
  isPending,
  onDelete,
  onRegenerate,
}: TrackCardProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleDelete = () => {
    if (!showDeleteConfirm) {
      setShowDeleteConfirm(true);
      setTimeout(() => setShowDeleteConfirm(false), 3000);
    } else {
      onDelete();
      setShowDeleteConfirm(false);
    }
  };

  // Extract features for display
  const features = track.features;
  const noteCount = features.note_count as number | undefined;
  const duration = features.duration_seconds as number | undefined;
  const noteDensity = features.note_density as number | undefined;

  return (
    <div
      className="group relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-white/8 to-white/4 p-5 shadow-xl backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:shadow-2xl"
      style={{
        animation: 'track-enter 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
      }}
    >
      {/* Pending Overlay */}
      {isPending && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="loading-callout">
            <span className="loading-dot" />
            Updating track…
          </div>
        </div>
      )}

      {/* Header */}
      <div className="mb-4 flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h4 className="text-lg font-semibold text-white">{track.instrument}</h4>
            <span className="rounded-full bg-rose-500/20 px-2 py-0.5 text-xs font-medium text-rose-300">
              {track.role}
            </span>
          </div>
          <p className="mt-1 font-mono text-xs text-slate-400">{track.id}</p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-slate-400 transition-all hover:bg-white/10 hover:text-slate-300"
            aria-label="Toggle details"
          >
            <svg
              className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <button
            onClick={onRegenerate}
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-slate-400 transition-all hover:bg-white/10 hover:text-rose-300 hover:rotate-180"
            aria-label="Regenerate"
            style={{ transition: 'all 0.3s ease' }}
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>

          <button
            onClick={handleDelete}
            className={`flex h-8 w-8 items-center justify-center rounded-lg transition-all ${
              showDeleteConfirm
                ? 'bg-red-500/90 text-white ring-2 ring-red-400'
                : 'bg-white/5 text-slate-400 hover:bg-red-500/20 hover:text-red-300'
            }`}
            aria-label={showDeleteConfirm ? 'Confirm delete' : 'Delete'}
          >
            {showDeleteConfirm ? (
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Feature Stats (Expanded) */}
      {isExpanded && (
        <div
          className="mb-4 grid grid-cols-3 gap-3 overflow-hidden rounded-xl bg-black/30 p-3"
          style={{
            animation: 'expand-down 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
          }}
        >
          {noteCount !== undefined && (
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-400">Notes</div>
              <div className="mt-1 font-mono text-lg font-semibold text-white">{noteCount}</div>
            </div>
          )}
          {duration !== undefined && (
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-400">Duration</div>
              <div className="mt-1 font-mono text-lg font-semibold text-white">{duration.toFixed(1)}s</div>
            </div>
          )}
          {noteDensity !== undefined && (
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-400">Density</div>
              <div className="mt-1 font-mono text-lg font-semibold text-white">
                {noteDensity.toFixed(2)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Audio Player */}
      <CustomAudioPlayer
        url={track.audio_url}
        label={track.id}
        variant="track"
      />

      {/* Download Links */}
      <div className="mt-3 flex items-center gap-3 text-xs">
        <a
          href={track.audio_url}
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
          MP3
        </a>
        <a
          href={track.midi_url}
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
          MIDI
        </a>
      </div>
    </div>
  );
}
