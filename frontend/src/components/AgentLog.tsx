import type { WsMessage } from '../types';

interface AgentLogProps {
  messages: WsMessage[];
  isGenerating: boolean;
}

function getIcon(type: string): string {
  switch (type) {
    case 'status': return '...';
    case 'plan': return '[P]';
    case 'tokens': return '[T]';
    case 'evaluation': return '[E]';
    case 'complete': return '[OK]';
    case 'error': return '[!]';
    default: return '[-]';
  }
}

function getColor(type: string): string {
  switch (type) {
    case 'status': return 'text-gray-400';
    case 'plan': return 'text-blue-400';
    case 'tokens': return 'text-green-400';
    case 'evaluation': return 'text-yellow-400';
    case 'complete': return 'text-emerald-400';
    case 'error': return 'text-red-400';
    default: return 'text-gray-400';
  }
}

function VerdictBadge({ verdict, score }: { verdict: string; score: number }) {
  const colors: Record<string, string> = {
    accept: 'bg-emerald-600',
    refine: 'bg-yellow-600',
    reject: 'bg-red-600',
  };

  return (
    <span className={`${colors[verdict] || 'bg-gray-600'} text-white text-xs px-2 py-0.5 rounded-full ml-2`}>
      {verdict} ({(score * 100).toFixed(0)}%)
    </span>
  );
}

export function AgentLog({ messages, isGenerating }: AgentLogProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
      <h3 className="text-lg font-semibold text-white mb-3">Agent Log</h3>

      <div className="space-y-2 max-h-80 overflow-y-auto text-sm font-mono">
        {messages.length === 0 && !isGenerating && (
          <p className="text-gray-500 italic">Waiting for composition...</p>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`${getColor(msg.type)} flex items-start gap-2`}>
            <span className="shrink-0 w-8 text-right">{getIcon(msg.type)}</span>
            <div className="flex-1">
              <span>{msg.message}</span>
              {msg.type === 'evaluation' && msg.verdict && msg.score !== undefined && (
                <VerdictBadge verdict={msg.verdict} score={msg.score} />
              )}
              {msg.type === 'evaluation' && msg.strengths && msg.strengths.length > 0 && (
                <div className="text-xs text-gray-500 mt-1">
                  + {msg.strengths.join(', ')}
                </div>
              )}
              {msg.type === 'evaluation' && msg.weaknesses && msg.weaknesses.length > 0 && (
                <div className="text-xs text-red-300/60 mt-0.5">
                  - {msg.weaknesses.join(', ')}
                </div>
              )}
              {msg.type === 'tokens' && msg.generation_time && (
                <span className="text-gray-500 ml-2">({msg.generation_time}s)</span>
              )}
            </div>
          </div>
        ))}

        {isGenerating && (
          <div className="text-blue-400 flex items-center gap-2 animate-pulse">
            <span className="w-8 text-right">...</span>
            <span>Working...</span>
          </div>
        )}
      </div>
    </div>
  );
}
