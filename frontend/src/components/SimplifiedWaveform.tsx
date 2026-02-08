import { useEffect, useRef } from 'react';
import type { MusicFeatures } from '../types';

interface SimplifiedWaveformProps {
  audioUrl: string;
  features?: MusicFeatures;
  variant: 'previous' | 'current';
}

export default function SimplifiedWaveform({ features, variant }: SimplifiedWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!features?.onset_density_curve || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw simplified waveform from onset_density_curve
    const curve = features.onset_density_curve;
    if (curve.length === 0) return;

    const maxValue = Math.max(...curve, 1); // Avoid division by zero
    const barWidth = canvas.width / curve.length;

    // Color based on variant
    const color = variant === 'current' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(156, 163, 175, 0.3)';
    ctx.fillStyle = color;

    curve.forEach((value, index) => {
      const barHeight = (value / maxValue) * canvas.height;
      const x = index * barWidth;
      const y = canvas.height - barHeight;

      ctx.fillRect(x, y, Math.max(barWidth - 1, 1), barHeight);
    });
  }, [features, variant]);

  if (!features?.onset_density_curve || features.onset_density_curve.length === 0) {
    return (
      <div className="relative h-16 overflow-hidden rounded-lg bg-black/20">
        <div className="flex h-full items-center justify-center text-xs text-gray-500">
          No waveform data
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-16 overflow-hidden rounded-lg bg-black/20">
      <canvas
        ref={canvasRef}
        width={800}
        height={64}
        className="h-full w-full"
      />
    </div>
  );
}
