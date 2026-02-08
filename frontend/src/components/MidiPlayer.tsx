interface MidiPlayerProps {
  midiUrl: string | null;
  textDescription: string;
  generationId: string;
}

export function MidiPlayer({ midiUrl, textDescription, generationId }: MidiPlayerProps) {
  if (!midiUrl) return null;

  const downloadUrl = `/api/outputs/${generationId}/midi`;

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
      <h3 className="text-lg font-semibold text-white mb-3">Generated Music</h3>

      {textDescription && (
        <p className="text-gray-300 text-sm mb-4 italic">
          "{textDescription}"
        </p>
      )}

      <div className="flex gap-3">
        <a
          href={downloadUrl}
          download={`${generationId}.mid`}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg
                     text-sm font-medium transition-colors inline-flex items-center gap-2"
        >
          Download MIDI
        </a>
      </div>

      <p className="text-gray-500 text-xs mt-3">
        Open the MIDI file in any music player or DAW to listen.
      </p>
    </div>
  );
}
