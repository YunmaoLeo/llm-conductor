import type { MusicFeatures } from '../types';

interface FeatureDisplayProps {
  features: MusicFeatures | null;
}

// General MIDI instrument names (subset)
const GM_INSTRUMENTS: Record<number, string> = {
  0: 'Piano', 1: 'Bright Piano', 2: 'Electric Grand', 4: 'Electric Piano',
  6: 'Harpsichord', 8: 'Celesta', 11: 'Vibraphone', 12: 'Marimba',
  22: 'Harmonica', 24: 'Nylon Guitar', 25: 'Steel Guitar', 26: 'Jazz Guitar',
  27: 'Clean Guitar', 28: 'Muted Guitar', 29: 'Overdrive Guitar',
  30: 'Distortion Guitar', 32: 'Acoustic Bass', 33: 'Fingered Bass',
  34: 'Picked Bass', 35: 'Fretless Bass', 40: 'Violin', 41: 'Viola',
  42: 'Cello', 43: 'Contrabass', 46: 'Harp', 48: 'String Ensemble',
  56: 'Trumpet', 57: 'Trombone', 60: 'French Horn', 61: 'Brass Section',
  64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax', 68: 'Oboe',
  71: 'Clarinet', 73: 'Flute', 80: 'Synth Lead',
};

function getInstrumentName(program: number): string {
  return GM_INSTRUMENTS[program] || `Program ${program}`;
}

function Bar({ value, max, color }: { value: number; max: number; color: string }) {
  const width = Math.min(100, (value / max) * 100);
  return (
    <div className="w-full bg-gray-700 rounded-full h-2">
      <div className={`${color} h-2 rounded-full`} style={{ width: `${width}%` }} />
    </div>
  );
}

export function FeatureDisplay({ features }: FeatureDisplayProps) {
  if (!features) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
      <h3 className="text-lg font-semibold text-white mb-3">Music Features</h3>

      <div className="space-y-3 text-sm">
        {/* Duration */}
        <div>
          <div className="flex justify-between text-gray-300 mb-1">
            <span>Duration</span>
            <span>{features.duration_seconds.toFixed(1)}s</span>
          </div>
          <Bar value={features.duration_seconds} max={120} color="bg-blue-500" />
        </div>

        {/* Note Density */}
        <div>
          <div className="flex justify-between text-gray-300 mb-1">
            <span>Note Density</span>
            <span>{features.note_density.toFixed(1)} notes/s</span>
          </div>
          <Bar value={features.note_density} max={30} color="bg-green-500" />
        </div>

        {/* Pitch Range */}
        <div>
          <div className="flex justify-between text-gray-300 mb-1">
            <span>Pitch Range</span>
            <span>MIDI {features.pitch_range[0]}-{features.pitch_range[1]}</span>
          </div>
          <Bar value={features.pitch_range[1] - features.pitch_range[0]} max={88} color="bg-purple-500" />
        </div>

        {/* Note Count */}
        <div className="flex justify-between text-gray-300">
          <span>Total Notes</span>
          <span className="text-white font-medium">{features.note_count}</span>
        </div>

        {/* Instruments */}
        <div>
          <span className="text-gray-300">Instruments</span>
          <div className="flex flex-wrap gap-1.5 mt-1">
            {features.instruments_used.map((prog) => (
              <span
                key={prog}
                className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-xs"
              >
                {getInstrumentName(prog)}
              </span>
            ))}
            {features.instrument_note_counts['-1'] && (
              <span className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-xs">
                Drums
              </span>
            )}
          </div>
        </div>

        {/* Activity Curve (mini sparkline) */}
        {features.onset_density_curve.length > 0 && (
          <div>
            <span className="text-gray-300">Activity</span>
            <div className="flex items-end gap-px mt-1 h-10">
              {features.onset_density_curve.map((v, i) => {
                const max = Math.max(...features.onset_density_curve, 1);
                const height = Math.max(2, (v / max) * 40);
                return (
                  <div
                    key={i}
                    className="bg-blue-500/60 rounded-t flex-1 min-w-0.5"
                    style={{ height: `${height}px` }}
                  />
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
