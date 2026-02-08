import { useEffect, useState } from 'react';
import { CustomAudioPlayer } from './CustomAudioPlayer';
import SimplifiedWaveform from './SimplifiedWaveform';
import FeatureComparisonTable from './FeatureComparisonTable';
import type { MusicFeatures } from '../types';

interface ComparisonViewProps {
  trackId: string;
  compositionId: string;
  currentVersion: {
    audio_url: string;
    features: MusicFeatures;
    version: number;
  };
  previousVersion: {
    audio_url: string;
    features?: MusicFeatures;
  };
}

export default function ComparisonView({
  trackId,
  compositionId,
  currentVersion,
  previousVersion,
}: ComparisonViewProps) {
  const [prevFeatures, setPrevFeatures] = useState<MusicFeatures | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch previous version features
  useEffect(() => {
    const fetchPreviousFeatures = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(
          `/api/tracks/${compositionId}/${trackId}/previous/features`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch previous version features');
        }

        const features = await response.json();
        setPrevFeatures(features);
        setError(null);
      } catch (err) {
        console.error('Error fetching previous features:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    fetchPreviousFeatures();
  }, [compositionId, trackId]);
  if (isLoading) {
    return (
      <div className="mt-4 rounded-lg border border-blue-500/30 bg-blue-500/5 p-4">
        <div className="flex items-center justify-center py-8 text-sm text-gray-400">
          <span className="loading-dot mr-2" />
          Loading previous version features...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/5 p-4">
        <div className="text-sm text-red-400">
          Failed to load previous version: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="mt-4 rounded-lg border border-blue-500/30 bg-blue-500/5 p-4">
      <h4 className="mb-3 text-sm font-semibold text-blue-400">
        Version Comparison
      </h4>

      {/* Side-by-side players */}
      <div className="grid grid-cols-2 gap-4">
        {/* Previous Version */}
        <div className="space-y-2">
          <div className="text-xs text-gray-400">
            Previous (v{currentVersion.version - 1})
          </div>
          <SimplifiedWaveform
            audioUrl={previousVersion.audio_url}
            features={prevFeatures || undefined}
            variant="previous"
          />
          <CustomAudioPlayer
            url={previousVersion.audio_url}
            variant="track"
          />
        </div>

        {/* Current Version */}
        <div className="space-y-2">
          <div className="text-xs text-blue-400">
            Current (v{currentVersion.version})
          </div>
          <SimplifiedWaveform
            audioUrl={currentVersion.audio_url}
            features={currentVersion.features}
            variant="current"
          />
          <CustomAudioPlayer
            url={currentVersion.audio_url}
            variant="track"
          />
        </div>
      </div>

      {/* Feature Comparison Table */}
      <FeatureComparisonTable
        previous={prevFeatures || undefined}
        current={currentVersion.features}
      />
    </div>
  );
}
