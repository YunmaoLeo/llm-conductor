import { useState, type FormEvent } from 'react';

interface ComposerPanelProps {
  onCompose: (intent: string) => void;
  isGenerating: boolean;
  onCancel: () => void;
}

export function ComposerPanel({ onCompose, isGenerating, onCancel }: ComposerPanelProps) {
  const [intent, setIntent] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (intent.trim() && !isGenerating) {
      onCompose(intent.trim());
    }
  };

  const presets = [
    'A gentle piano melody',
    'An energetic rock song with drums and guitar',
    'A melancholy jazz piece with saxophone',
    'A cheerful orchestral march',
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <h2 className="text-xl font-semibold text-white mb-4">Compose</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={intent}
          onChange={(e) => setIntent(e.target.value)}
          placeholder="Describe the music you want to create..."
          className="w-full h-28 bg-gray-700 text-white rounded-lg p-3 resize-none
                     placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isGenerating}
        />

        <div className="flex gap-2">
          {isGenerating ? (
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2.5 px-4
                         rounded-lg font-medium transition-colors"
            >
              Cancel
            </button>
          ) : (
            <button
              type="submit"
              disabled={!intent.trim()}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600
                         disabled:cursor-not-allowed text-white py-2.5 px-4 rounded-lg
                         font-medium transition-colors"
            >
              Compose
            </button>
          )}
        </div>
      </form>

      <div className="mt-4">
        <p className="text-gray-400 text-sm mb-2">Quick presets:</p>
        <div className="flex flex-wrap gap-2">
          {presets.map((preset) => (
            <button
              key={preset}
              onClick={() => setIntent(preset)}
              disabled={isGenerating}
              className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300
                         px-3 py-1.5 rounded-full transition-colors disabled:opacity-50"
            >
              {preset}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
