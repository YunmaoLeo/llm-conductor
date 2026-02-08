import { useEffect, useRef, useState } from 'react';

interface CustomAudioPlayerProps {
  url: string;
  label?: string;
  variant?: 'track' | 'mix';
}

export function CustomAudioPlayer({
  url,
  label: _label,
  variant = 'track',
}: CustomAudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressBarRef = useRef<HTMLDivElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.7);
  const [isLoading, setIsLoading] = useState(true);
  const [showVolumeSlider, setShowVolumeSlider] = useState(false);

  // Initialize audio element
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadStart = () => setIsLoading(true);
    const handleCanPlay = () => setIsLoading(false);
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleDurationChange = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('loadstart', handleLoadStart);
    audio.addEventListener('canplay', handleCanPlay);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('durationchange', handleDurationChange);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('loadstart', handleLoadStart);
      audio.removeEventListener('canplay', handleCanPlay);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('durationchange', handleDurationChange);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);


  // Handle volume
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = volume;
  }, [volume]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(console.error);
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    const progressBar = progressBarRef.current;
    if (!audio || !progressBar || !duration) return;

    const rect = progressBar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    audio.currentTime = percentage * duration;
  };

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="relative">
      <audio ref={audioRef} src={url} preload="metadata" />

      {/* Progress Bar */}
      <div className="relative mb-3">
        <div
          ref={progressBarRef}
          onClick={handleSeek}
          className="group relative h-2 cursor-pointer overflow-hidden rounded-full bg-white/10 transition-all hover:h-2.5"
        >
          {/* Filled portion */}
          <div
            className="absolute left-0 top-0 h-full bg-gradient-to-r from-rose-500 to-rose-600 transition-all"
            style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
          />
          {/* Hover indicator */}
          <div className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100">
            <div className="h-full bg-white/5" />
          </div>
        </div>
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="loading-mini text-xs">
              <span className="loading-dot" />
              Loading audio…
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        {/* Play/Pause Button */}
        <button
          onClick={togglePlay}
          disabled={isLoading}
          className="group flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-rose-500 to-rose-600 shadow-lg shadow-rose-500/25 transition-all hover:scale-105 hover:shadow-rose-500/40 active:scale-95 disabled:opacity-50 disabled:hover:scale-100"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg className="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path d="M6 4h2v12H6V4zm6 0h2v12h-2V4z" />
            </svg>
          ) : (
            <svg className="h-4 w-4 translate-x-0.5 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path d="M6.3 4.1c-.2-.1-.4-.1-.6 0-.2.1-.3.3-.3.5v10.8c0 .2.1.4.3.5.1.1.2.1.3.1.1 0 .2 0 .3-.1l8.5-5.4c.2-.1.3-.3.3-.5s-.1-.4-.3-.5l-8.5-5.4z" />
            </svg>
          )}
        </button>

        {/* Time Display */}
        <div className="flex items-center gap-2 text-xs font-mono tabular-nums text-slate-300">
          <span>{formatTime(currentTime)}</span>
          <span className="text-slate-500">/</span>
          <span>{formatTime(duration)}</span>
        </div>


        {/* Volume Control */}
        <div
          className="relative ml-auto"
          onMouseEnter={() => setShowVolumeSlider(true)}
          onMouseLeave={() => setShowVolumeSlider(false)}
        >
          <button
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-slate-400 transition-colors hover:bg-white/10 hover:text-slate-300"
            aria-label="Volume"
          >
            {volume === 0 ? (
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM12.293 7.293a1 1 0 011.414 0L15 8.586l1.293-1.293a1 1 0 111.414 1.414L16.414 10l1.293 1.293a1 1 0 01-1.414 1.414L15 11.414l-1.293 1.293a1 1 0 01-1.414-1.414L13.586 10l-1.293-1.293a1 1 0 010-1.414z" />
              </svg>
            ) : volume < 0.5 ? (
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414z" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM12.293 7.293a1 1 0 011.414 0L15 8.586l1.293-1.293a1 1 0 111.414 1.414L16.414 10l1.293 1.293a1 1 0 01-1.414 1.414L15 11.414l-1.293 1.293a1 1 0 01-1.414-1.414L13.586 10l-1.293-1.293a1 1 0 010-1.414z" />
              </svg>
            )}
          </button>

          {showVolumeSlider && (
            <div className="absolute bottom-full right-0 mb-2 rounded-lg bg-slate-900/95 p-3 shadow-xl backdrop-blur-sm">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="volume-slider h-20 w-6"
                style={{ writingMode: 'vertical-lr' } as React.CSSProperties}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
