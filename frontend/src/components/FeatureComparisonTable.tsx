import type { MusicFeatures } from '../types';

interface FeatureComparisonTableProps {
  previous?: MusicFeatures;
  current: MusicFeatures;
}

export default function FeatureComparisonTable({ previous, current }: FeatureComparisonTableProps) {
  if (!previous) {
    return (
      <div className="mt-4 text-xs text-gray-500">
        No previous version data available
      </div>
    );
  }

  const calculateChange = (prev: number, curr: number): string => {
    if (prev === 0) return curr > 0 ? '+∞%' : '0%';
    const diff = curr - prev;
    const percent = ((diff / prev) * 100).toFixed(1);
    if (diff > 0) return `+${percent}%`;
    if (diff < 0) return `${percent}%`;
    return '0%';
  };

  const getChangeColor = (prev: number, curr: number): string => {
    if (curr > prev) return 'text-green-400';
    if (curr < prev) return 'text-red-400';
    return 'text-gray-400';
  };

  const comparisons = [
    {
      label: 'Note Count',
      prev: previous.note_count,
      curr: current.note_count,
    },
    {
      label: 'Density (notes/sec)',
      prev: previous.note_density,
      curr: current.note_density,
    },
    {
      label: 'Duration (sec)',
      prev: previous.duration_seconds,
      curr: current.duration_seconds,
    },
    {
      label: 'Pitch Range',
      prev: previous.pitch_range[1] - previous.pitch_range[0],
      curr: current.pitch_range[1] - current.pitch_range[0],
    },
  ];

  return (
    <div className="mt-4">
      <div className="mb-2 text-xs font-semibold text-gray-400">Feature Changes</div>
      <div className="grid grid-cols-4 gap-2">
        {comparisons.map((comp) => (
          <div key={comp.label} className="rounded-lg bg-black/20 p-2">
            <div className="mb-1 text-[10px] text-gray-500">{comp.label}</div>
            <div className="flex items-baseline gap-2">
              <span className="text-xs text-gray-400">{comp.prev.toFixed(1)}</span>
              <span className="text-xs">→</span>
              <span className="text-xs font-semibold text-white">{comp.curr.toFixed(1)}</span>
            </div>
            <div className={`mt-1 text-[10px] ${getChangeColor(comp.prev, comp.curr)}`}>
              {calculateChange(comp.prev, comp.curr)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
