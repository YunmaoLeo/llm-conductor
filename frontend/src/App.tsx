import { ComposerPanel } from './components/ComposerPanel';
import { AgentLog } from './components/AgentLog';
import { MidiPlayer } from './components/MidiPlayer';
import { FeatureDisplay } from './components/FeatureDisplay';
import { useGeneration } from './hooks/useGeneration';

function App() {
  const { status, messages, result, error, compose, cancel, isGenerating } = useGeneration();

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-700 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">LLM-Conductor</h1>
            <p className="text-gray-400 text-sm">Agent-based Symbolic Music Orchestration</p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${
              status === 'idle' ? 'bg-gray-500' :
              isGenerating ? 'bg-yellow-400 animate-pulse' :
              status === 'complete' ? 'bg-emerald-400' :
              status === 'error' ? 'bg-red-400' : 'bg-gray-500'
            }`} />
            <span className="text-gray-400 text-sm capitalize">{status}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Composer + Player */}
          <div className="lg:col-span-2 space-y-6">
            <ComposerPanel
              onCompose={compose}
              isGenerating={isGenerating}
              onCancel={cancel}
            />

            {error && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">
                {error}
              </div>
            )}

            {result && (
              <MidiPlayer
                midiUrl={result.midiUrl}
                textDescription={result.textDescription}
                generationId={result.generationId}
              />
            )}

            {result?.features && (
              <FeatureDisplay features={result.features} />
            )}
          </div>

          {/* Right Column: Agent Log */}
          <div>
            <AgentLog messages={messages} isGenerating={isGenerating} />

            {result && (
              <div className="mt-4 bg-gray-800 rounded-lg p-4 shadow-lg">
                <h3 className="text-lg font-semibold text-white mb-2">Summary</h3>
                <div className="text-sm text-gray-300 space-y-1">
                  <p>Iterations: {result.iterations}</p>
                  {result.features && (
                    <>
                      <p>Notes: {result.features.note_count}</p>
                      <p>Duration: {result.features.duration_seconds.toFixed(1)}s</p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
